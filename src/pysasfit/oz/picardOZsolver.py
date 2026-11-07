# -*- coding: utf-8 -*-
#calculations
import numpy as np

#Base class
from .oZsolver import OZsolver
#This class inherits all that is needed to run the 
#iteration and to communicate with clients from OZsolver
class PicardOZsolver(OZsolver):
    #PICARD NEEDS THE ROOM, where the accelerated solvers do not. Measured
    #on hard spheres under PY it takes 336 iterations at phi = 0.30 against
    #17 to 60 for SUNDIALS KIN_FP, and the count grows geometrically with
    #density -- 41, 49, 60, 74, ... 946 in steps of 0.02 -- until it stops
    #converging altogether above about phi = 0.42. So 600 would cut off
    #solves that would have succeeded; 3000 does not, while still being a
    #long way below the 6000 and 8000 that callers used to pass.
    #
    #RAISED TO 5000 when Picard / Mann became the damped second verifier: at
    #hard spheres PY phi = 0.53 it needs about 4170/a iterations, so with
    #a = 0.5 and N = 3000 (limit 6000) it stopped at 6000 of the 8334
    #needed, and with Biggs-Andrews also failing there no verifier converged.
    #N = 5000 gives 10000 at a = 0.5. It is a limit, so it costs nothing on a
    #solve that converges.
    DEFAULT_MAX_ITERATIONS = 5000

    def __init__(self, port, **kwargs):
      #First run constructor of super class
      OZsolver.__init__(self, port, **kwargs)

    #EARLY STOP (2026-11-07). Two checks end an iteration that cannot give a
    #solution, instead of running into the iteration limit:
    #
    #  * UNPHYSICAL: every PHYS_CHECK_EVERY steps after PHYS_WARMUP steps the
    #    pair correlation g = 1 + Gamma + c of the current iterate is formed
    #    (one closure evaluation, no transform); when min g < PHYS_GMIN at two
    #    consecutive checks the closure has no physical solution here. Seen for
    #    PY, MSA, SMSA, Verlet, DH, CG on the strongly charged polydisperse
    #    Yukawa test state (Z = 200): g reached -700 ... -2500 while the
    #    residual still fell slowly; MDIIS fails there as well.
    #  * STALLED: the residual decay rate over the last STALL_WINDOW steps,
    #    extrapolated, needs more than STALL_FACTOR times the remaining
    #    iterations to reach the convergence criterion (checked twice in a row).
    #    Such a run would end at the limit unconverged anyway.
    #
    #A converged iteration is unaffected (the checks only ever stop a run that
    #has not converged); EARLY_STOP = False restores the former behaviour.
    #The reason is kept in self.stopReason and reported by ozLib.solve().
    #Preferences: oz.solvers.picard_early_stop, oz.solvers.picard_gmin.
    EARLY_STOP = True
    PHYS_GMIN = -0.5
    PHYS_CHECK_EVERY = 50
    PHYS_WARMUP = 200
    STALL_WINDOW = 200
    STALL_FACTOR = 3.0

    def _minRDF(self, x):
        """min g(r) = min(1 + Gamma + c) of iterate x (all partials for a mixture); NaN if not available."""
        try:
            if not getattr(self, 'doUseGammaFixPointOperator', True):
                return float('nan')
            G = self.unpackPairs(x) if self.isMulticomponent() else x
            g = G + self.update_c(G) + 1.0
            g = g[np.isfinite(g)]
            return float(g.min()) if g.size else float('nan')
        except Exception:
            return float('nan')

    #define the specific iteration schema for this class    
    def picardIteration(self):
      #MANN / KRASNOSELSKII ITERATION.
      #
      #  x_{n+1} = (1 - alpha) x_n + alpha T(x_n)
      #
      #Plain Picard is the alpha = 1 case. The averaged form is due to Mann
      #(1953); with constant alpha it is usually called Krasnoselskii, and
      #the Krasnoselskii-Mann theorem is what gives convergence for
      #NONEXPANSIVE maps, where undamped Picard need not converge at all.
      #
      #alpha defaults to 1.0, so existing behaviour and every recorded
      #regression value are unchanged -- verified: HardSphere PY g_max
      #2.35611803 and LennardJones HNC 2.16359468, both exact. Set
      #self.mannAlpha to damp.
      #
      #This matters because Picard is the default VERIFIER in ozLib.solve(),
      #chosen for independence -- it is the only solver here with no
      #acceleration and no history vectors, so it cannot share an
      #acceleration failure mode with the primary. Undamped it is also the
      #most fragile: it returns NaN for Lennard-Jones at epsilon = 0.8,
      #phi = 0.3 under a type-4 transform, where alpha = 0.5 converges to the
      #correct 2.164596. A verifier that fails is nearly as unhelpful as one
      #that agrees wrongly.
      #
      #Set from ozLib.solve(mannAlpha=...), the "damping a" field of the OZ
      #solver, polydisperse and mixture tabs, or directly as
      #self.mannAlpha. The default stays undamped so nothing changes
      #silently.
      alpha = float(getattr(self, 'mannAlpha', 1.0))
      if not (0.0 < alpha <= 1.0):
          raise ValueError(f"mannAlpha must be in (0, 1], got {alpha}")
      #NOT CONVERGED UNTIL PROVEN OTHERWISE.
      #
      #Exhausting the iteration limit used to print a warning and return the
      #last iterate regardless, with nothing a caller could test. That is how
      #BPGG alpha = 0.5, phi = 0.4 produced a "solution" whose residual was
      #|F| = 21.3. Callers should check `converged` (and, better, recompute
      #the residual themselves -- a flag is a claim, a residual is evidence).
      self.converged = False
      self.finalRelativeProgress = float('inf')
      #ITERATION LIMIT SCALED BY 1/a. A damped step moves only a fraction a
      #of the way to T(x), so Mann's iteration needs about 1/a times the
      #iterations of an undamped one (measured: HS/PY phi = 0.53 converges
      #after about 4170/a steps for every a <= 0.6). The configured limit N
      #(numberOfIterations: Preferences -> oz.solvers.max_iterations_picard,
      #or the maxIterations a caller passed) is therefore the limit for
      #a = 1, and the iteration runs up to ceil(N/a) steps. a = 1 gives N.
      self.iterationLimit = self.mannIterationLimit(self.numberOfIterations, alpha)
      self.iterations = 0
      self.stopReason = None
      early = bool(getattr(self, 'EARLY_STOP', True))
      nBad = nStall = 0
      logRes = []
      #Start iteration
      x = self.x_0
      #Converged if relative progress is less than self.convergenceCriterion
      i = 0; previousNorm = 0.0; xPrevious = None
      while  i < self.iterationLimit and not self.isInterrupted:
          xPrevious = x
          xNew = self.fixPointOperator(x)
          x = xNew if alpha == 1.0 else (1.0 - alpha)*x + alpha*xNew
          norm = np.linalg.norm(x)
          #CONVERGENCE ON THE INCREMENT, not on successive NORMS.
          #
          #This previously read
          #    relativeProgress = abs(previousNorm - norm)/(norm + eps)
          #which compares ||x_n|| with ||x_{n-1}||. Two DIFFERENT vectors of
          #similar norm pass that test, so an iterate still moving but whose
          #norm has momentarily plateaued is reported as converged. That is
          #not hypothetical: at BPGG alpha = 0.5, phi = 0.4 it returned
          #g_max = 3.0039 as a converged solution when the residual there was
          #|F| = 19.4 -- not a fixed point at all, while the true root is
          #5.2528. The error was only found by evaluating |F| at the claimed
          #solution, which nothing in the pipeline was doing.
          #
          #CONVERGENCE ON THE RESIDUAL ||T(x) - x||, not on the damped
          #increment. With damping the increment is a*(T(x) - x): testing
          #the increment would declare convergence 1/a times too early (at
          #a = 0.1 a residual ten times the criterion would pass). For
          #a = 1 the two are identical, so plain Picard and every recorded
          #regression value are unchanged.
          relativeProgress = (np.linalg.norm(xNew - xPrevious)
                              / (norm + np.finfo(float).eps))
          previousNorm = norm
          self.finalRelativeProgress = float(relativeProgress)
          if relativeProgress < self.convergenceCriterion: #Defined in super class
              #pass
              self.converged = True
              print(self._name(alpha), "converged after", i, "steps")
              self.iterations = i
              break
          if early:
              if not np.isfinite(relativeProgress):
                  self.stopReason = "the iteration diverged (non-finite residual)"
                  break
              logRes.append(np.log(max(relativeProgress, 1e-300)))
              if i >= self.PHYS_WARMUP and i % self.PHYS_CHECK_EVERY == 0:
                  gmin = self._minRDF(x)
                  nBad = nBad + 1 if gmin < self.PHYS_GMIN else 0
                  if nBad >= 2:
                      self.stopReason = (f"unphysical iterate: min g(r) = {gmin:.3g} < {self.PHYS_GMIN:g} "
                                         f"at two checks; the closure has no physical solution here")
                      break
                  W = self.STALL_WINDOW
                  if len(logRes) > W:
                      slope = (logRes[-1] - logRes[-1 - W]) / W          # ln residual per step
                      need = (np.log(self.convergenceCriterion) - logRes[-1]) / slope if slope < 0 else np.inf
                      nStall = nStall + 1 if need > self.STALL_FACTOR * (self.iterationLimit - i) else 0
                      if nStall >= 2:
                          self.stopReason = (f"stalled: at the current decay rate the residual would need about "
                                             f"{need:.3g} more steps (limit {self.iterationLimit})")
                          break
          i += 1
          
      #endwhile
      if not self.converged:
          self.iterations = i
      if self.stopReason:
          print(self._name(alpha), f"stopped after {i} steps:", self.stopReason)
      if i == self.iterationLimit:
          print(self._name(alpha), "did not converge after", self.iterationLimit, "steps"
                + ("" if alpha == 1.0 else f" (limit {self.numberOfIterations}/a)"))

      self.derivePhysicalQuantitiesFromFixpoint(x)
    @staticmethod
    def mannIterationLimit(n, alpha):
        """Iteration limit of the Picard / Mann iteration: ceil(n / a) for the
        configured limit n (the limit for a = 1) and the damping 0 < a <= 1."""
        alpha = float(alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"mannAlpha must be in (0, 1], got {alpha}")
        return int(np.ceil(int(n) / alpha - 1e-9))

    @staticmethod
    def _name(alpha):
        return "Picard" if alpha == 1.0 else f"Picard/Mann (a = {alpha:g})"

    #implement the abstract method such that this class can be instantiated
    def solve(self):
      self.picardIteration()
