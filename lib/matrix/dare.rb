# frozen_string_literal: false
class Matrix
  # The result of Matrix.solve_discrete_are.
  #
  # Attributes:
  # * +x+:                  the computed stabilizing solution of the Riccati
  #                         equation (a +Matrix+, or +nil+ if the computation
  #                         broke down before any finite iterate was produced).
  # * +gain+:               the optimal state feedback gain +K+ such that
  #                         <tt>u = -K * x</tt> and the closed-loop matrix is
  #                         <tt>A - B * K</tt> (+nil+ whenever +x+ is +nil+).
  # * +closed_loop_matrix+: <tt>A - B * gain</tt> (+nil+ whenever +x+ is +nil+).
  # * +failure_reason+:     +nil+ on success, otherwise a +Symbol+ describing
  #                         why no trustworthy solution could be produced; see
  #                         Matrix.solve_discrete_are for the possible values.
  # * +diagnostics+:        a +Hash+ of quantities that indicate how much the
  #                         result can be trusted; see Matrix.solve_discrete_are.
  #
  # Note that +x+ and +gain+ are only meaningful when <tt>result.ok?</tt>;
  # on failure they hold the best (untrustworthy) approximation found, to aid
  # debugging.
  class DiscreteAREResult
    attr_reader :x, :gain, :closed_loop_matrix, :failure_reason, :diagnostics

    def initialize(x:, gain:, closed_loop_matrix:, failure_reason:, diagnostics:)
      @x = x
      @gain = gain
      @closed_loop_matrix = closed_loop_matrix
      @failure_reason = failure_reason
      @diagnostics = diagnostics.freeze
      freeze
    end

    # Returns +true+ when a trustworthy stabilizing solution was computed.
    def ok?
      @failure_reason.nil?
    end

    def to_s
      "#<#{self.class} #{ok? ? "ok" : "failed: #{failure_reason}"}>"
    end
    alias_method :inspect, :to_s
  end

  # Relative change in the iterates below which the doubling iteration is
  # considered converged.
  DARE_CONVERGENCE_TOLERANCE = 1e-12
  # Closed-loop eigenvalues whose modulus is within this distance of 1 cannot
  # be separated reliably from their reciprocal (symplectic) counterparts.
  DARE_UNIT_CIRCLE_BAND = 1e-6
  # Separation min |1 - conj(li) * lj| of the closed-loop spectrum below which
  # the stable invariant subspace is considered ill-conditioned.
  DARE_SEPARATION_TOLERANCE = 1e-4
  # Relative residual above which the computed solution is rejected.
  DARE_RESIDUAL_TOLERANCE = 1e-8
  private_constant :DARE_CONVERGENCE_TOLERANCE, :DARE_UNIT_CIRCLE_BAND,
                   :DARE_SEPARATION_TOLERANCE, :DARE_RESIDUAL_TOLERANCE

  # Solves the discrete-time algebraic Riccati equation (DARE)
  #
  #   X = A'XA - A'XB (R + B'XB)^-1 B'XA + Q
  #
  # for the stabilizing solution +X+, where +A+ is n x n, +B+ is n x m,
  # +Q+ is a symmetric positive semidefinite n x n matrix and +R+ is a
  # symmetric positive definite m x m matrix.  This is the equation behind
  # the discrete-time LQR optimal state feedback <tt>u = -K x</tt> with
  # <tt>K = (R + B'XB)^-1 B'XA</tt>.
  #
  # The equation is solved with the structure-preserving doubling algorithm
  # (SDA), a quadratically convergent iteration that only requires matrix
  # products and inversions; no external numerical library is needed.  The
  # iteration stops when the relative change of the iterates drops below
  # 1e-12 (at most +max_iterations+ sweeps), never after a fixed number of
  # steps.  Entries are converted to Float and the computation is carried
  # out in double precision.
  #
  # Returns a Matrix::DiscreteAREResult.  Instead of raising on numerical
  # failure, the result carries a +failure_reason+ and a +diagnostics+ hash
  # so the caller can judge how much the answer can be trusted:
  #
  #   result = Matrix.solve_discrete_are(a, b, q, r)
  #   if result.ok?
  #     k = result.gain            # u = -k * x stabilizes the system
  #   else
  #     warn result.failure_reason # e.g. :unstabilizable
  #     warn result.diagnostics[:message]
  #   end
  #
  # Possible failure reasons:
  # [:non_convergence]
  #   The doubling iteration did not converge (divergence, breakdown, or too
  #   slow).  Typical causes: (A, B) not stabilizable, (Q, A) not detectable,
  #   or symplectic eigenvalues lying on the unit circle.
  # [:unstabilizable]
  #   A solution was found but the closed loop <tt>A - B*K</tt> is not
  #   asymptotically stable (or +X+ is not positive semidefinite), so no
  #   stabilizing solution exists.
  # [:spectrum_near_unit_circle]
  #   Closed-loop eigenvalues lie within 1e-6 of the unit circle; the
  #   symplectic spectrum cannot be classified into stable/unstable parts
  #   reliably, so the solution cannot be trusted.
  # [:ill_conditioned_subspace]
  #   The stable invariant subspace is ill-conditioned (separation of the
  #   closed-loop spectrum below 1e-4); small perturbations of the data cause
  #   large changes in +X+.
  # [:large_residual]
  #   The computed +X+ does not satisfy the Riccati equation accurately
  #   (relative residual above 1e-8).
  #
  # The +diagnostics+ hash contains:
  # [:converged]                  whether the iteration converged
  # [:iterations]                 number of doubling sweeps performed
  # [:relative_residual]          ||F(X) - X||_F / (||X||_F + ||F(X)||_F)
  # [:closed_loop_eigenvalues]    eigenvalues of A - B*K (Float or Complex)
  # [:closed_loop_spectral_radius] largest eigenvalue modulus
  # [:stability_margin]           1 - spectral radius (positive when stable)
  # [:subspace_separation]        min |1 - conj(li)*lj| over closed-loop
  #                               eigenvalues; its magnitude is the reciprocal
  #                               condition estimate of the invariant subspace
  # [:message]                    human readable summary of the outcome
  #
  # Raises ArgumentError if the dimensions do not match, if +Q+ is not
  # symmetric positive semidefinite, or if +R+ is not symmetric positive
  # definite.
  def self.solve_discrete_are(a, b, q, r, max_iterations: 100)
    a, b, q, r = dare_checked_inputs(a, b, q, r)
    n = a.row_count

    # Structure-preserving doubling algorithm:
    #   A+ = A (I + G H)^-1 A
    #   G+ = G + A (I + G H)^-1 G A'
    #   H+ = H + A' (I + G H)^-T H A
    # starting from G = B R^-1 B', H = Q; H converges to the stabilizing X.
    g = b * r.inverse * b.t
    ak, hk = a, q
    converged = false
    breakdown = false
    iterations = 0
    while iterations < max_iterations
      begin
        w = (Matrix.I(n) + g * hk).inverse
      rescue ErrNotRegular
        breakdown = true
        break
      end
      ak_next = ak * w * ak
      g = dare_symmetrize(g + ak * w * g * ak.t)
      hk_next = dare_symmetrize(hk + ak.t * w.t * hk * ak)
      delta = dare_frobenius_norm(hk_next - hk) / (1.0 + dare_frobenius_norm(hk_next))
      ak, hk = ak_next, hk_next
      iterations += 1
      break if (converged = delta <= DARE_CONVERGENCE_TOLERANCE)
      break unless delta.finite?
    end

    x = hk.all? { |e| e.finite? } ? hk : nil
    gain = closed_loop = nil
    residual = eigs = rho = separation = nil
    if x
      gain = (r + b.t * x * b).inverse * (b.t * x * a)
      closed_loop = a - b * gain
      eigs = closed_loop.eigen.eigenvalues
      rho = eigs.map(&:abs).max
      separation = eigs.flat_map { |li| eigs.map { |lj| (1 - li.conj * lj).abs } }.min
      riccati = a.t * x * a - (a.t * x * b) * gain + q
      residual = dare_frobenius_norm(riccati - x) /
                 (dare_frobenius_norm(x) + dare_frobenius_norm(riccati))
    end

    failure_reason, message = dare_assess(
      converged, breakdown, max_iterations, x, rho, separation, residual
    )

    DiscreteAREResult.new(
      x: x,
      gain: gain,
      closed_loop_matrix: closed_loop,
      failure_reason: failure_reason,
      diagnostics: {
        converged: converged,
        iterations: iterations,
        relative_residual: residual,
        closed_loop_eigenvalues: eigs,
        closed_loop_spectral_radius: rho,
        stability_margin: rho && 1.0 - rho,
        subspace_separation: separation,
        message: message,
      }
    )
  end

  class << self
    private

    def dare_checked_inputs(a, b, q, r)
      { "A" => a, "B" => b, "Q" => q, "R" => r }.each do |name, m|
        raise ArgumentError, "#{name} must be a Matrix" unless m.is_a?(Matrix)
      end
      raise ArgumentError, "A must be square" unless a.square?
      n = a.row_count
      raise ArgumentError, "A must not be empty" if n.zero?
      raise ArgumentError, "B must have as many rows as A (#{n})" unless b.row_count == n
      m = b.column_count
      raise ArgumentError, "B must have at least one column" if m.zero?
      unless q.square? && q.row_count == n
        raise ArgumentError, "Q must be square of the same order as A (#{n})"
      end
      unless r.square? && r.row_count == m
        raise ArgumentError, "R must be square of order #{m} (columns of B)"
      end

      begin
        a, b, q, r = [a, b, q, r].map { |mx| mx.map { |e| Float(e) } }
      rescue TypeError, RangeError
        raise ArgumentError, "only real numeric matrix entries are supported"
      end

      q = dare_force_symmetric(q, "Q")
      r = dare_force_symmetric(r, "R")

      q_eigs = q.eigen.eigenvalues
      q_scale = [q_eigs.map(&:abs).max, 1.0].max
      if q_eigs.min < -1e-10 * q_scale
        raise ArgumentError, "Q must be positive semidefinite"
      end
      r_eigs = r.eigen.eigenvalues
      r_scale = [r_eigs.map(&:abs).max, 1.0].max
      if r_eigs.min <= 1e-14 * r_scale
        raise ArgumentError, "R must be positive definite"
      end

      [a, b, q, r]
    end

    def dare_force_symmetric(m, name)
      return m if m.symmetric?
      skew = dare_frobenius_norm(m - m.t)
      if skew <= 1e-10 * [dare_frobenius_norm(m), 1.0].max
        (m + m.t) / 2
      else
        raise ArgumentError, "#{name} must be symmetric"
      end
    end

    def dare_frobenius_norm(m)
      Math.sqrt(m.each.inject(0.0) { |sum, e| sum + e * e })
    end

    def dare_symmetrize(m)
      (m + m.t) / 2
    end

    def dare_assess(converged, breakdown, max_iterations, x, rho, separation, residual)
      unless converged
        message =
          if breakdown
            "doubling iteration broke down: I + G*H became singular; " +            "the symplectic spectrum likely has eigenvalues on the unit circle"
          else
            "doubling iteration did not converge within #{max_iterations} " +            "iterations; (A, B) may be unstabilizable, (Q, A) undetectable, " +            "or the symplectic spectrum may lie on the unit circle"
          end
        return [:non_convergence, message]
      end

      x_min_eig = x.eigen.eigenvalues.min
      if x_min_eig < -DARE_RESIDUAL_TOLERANCE * [dare_frobenius_norm(x), 1.0].max
        return [:unstabilizable,
                "computed solution is not positive semidefinite; no " +                "stabilizing solution exists (check stabilizability of (A, B) " +                "and detectability of (Q, A))"]
      end

      if rho > 1.0 + DARE_UNIT_CIRCLE_BAND
        return [:unstabilizable,
                format("no stabilizing solution: closed-loop spectral radius " +                       "%g >= 1; the system is unstabilizable or undetectable", rho)]
      end
      if (1.0 - rho).abs <= DARE_UNIT_CIRCLE_BAND
        return [:spectrum_near_unit_circle,
                format("closed-loop spectral radius %g is within %g of the " +                       "unit circle; the symplectic spectrum cannot be " +                       "classified reliably", rho, DARE_UNIT_CIRCLE_BAND)]
      end
      if separation <= DARE_SEPARATION_TOLERANCE
        return [:ill_conditioned_subspace,
                format("stable invariant subspace is ill-conditioned " +                       "(separation %g); small perturbations of A, B, Q, R " +                       "cause large errors in X", separation)]
      end
      if residual > DARE_RESIDUAL_TOLERANCE
        return [:large_residual,
                format("relative residual %g exceeds %g; X does not satisfy " +                       "the Riccati equation accurately", residual, DARE_RESIDUAL_TOLERANCE)]
      end
      [nil, format("converged; relative residual %g, closed-loop spectral " +                   "radius %g", residual, rho)]
    end
  end

end
