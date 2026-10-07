# frozen_string_literal: false
class Matrix
  # Williamson decomposition of a real symmetric positive definite matrix.
  #
  # For an even-sized matrix +A+, returns a symplectic transform +S+ and
  # diagonal +D+ satisfying:
  #
  #   A = S * D * S.transpose
  #   S.transpose * J * S = J
  #
  # Coordinates use [q1, p1, q2, p2, ...] and +J+ is the corresponding
  # block-diagonal canonical symplectic matrix. Each distinct value in +D+
  # occurs twice. Symplectic eigenvalues are sorted nondecreasingly. The
  # constructed basis is reproducible, but bases inside repeated or nearly
  # repeated symplectic eigenspaces are not unique.
  class WilliamsonDecomposition
    include ExceptionForMatrix

    attr_reader :symplectic_eigenvalues, :symplectic_transform
    alias_method :s, :symplectic_transform

    def initialize(matrix)
      raise TypeError, "Expected Matrix but got #{matrix.class}" unless matrix.is_a?(Matrix)
      size = matrix.row_count
      unless matrix.square?
        raise ErrDimensionMismatch, "Williamson decomposition requires a square matrix"
      end
      unless size.positive? && size.even?
        raise ErrDimensionMismatch, "Williamson decomposition requires a positive even-sized matrix"
      end
      unless matrix.real?
        raise TypeError, "Williamson decomposition supports real matrices only"
      end

      entries = matrix.to_a
      scale = entries.flatten.map(&:abs).max.to_f
      if scale.zero? || !scale.finite?
        raise ErrNotPositiveDefinite, "Matrix is not positive definite"
      end

      tolerance = 64 * size * Float::EPSILON
      unless symmetric?(entries, scale, tolerance)
        raise ErrNotSymmetric, "Matrix is not symmetric within numerical precision"
      end

      a = Array.new(size) do |i|
        Array.new(size) { |j| (entries[i][j].to_f + entries[j][i].to_f) / (2.0 * scale) }
      end
      lower, diagonal, permutation = pivoted_ldl(a)
      w = Array.new(size) do |i|
        Array.new(size) do |j|
          j <= i ? lower[i][j] * Math.sqrt(diagonal[j]) : 0.0
        end
      end

      j = canonical_j(size / 2)
      j_permuted = Array.new(size) do |i|
        Array.new(size) { |k| j[permutation[i]][permutation[k]] }
      end
      k = skew_part(multiply_transpose_left(w, j_permuted))
      q_columns, normalized_eigenvalues = skew_singular_vectors(k, tolerance)
      @symplectic_eigenvalues = normalized_eigenvalues.each_slice(2).map { |value, _| value * scale }

      inverse_sqrt_values = normalized_eigenvalues.map { |value| 1.0 / Math.sqrt(value) }
      q = Matrix.send(:new, q_columns.transpose)
      d_inverse = Matrix.diagonal(*inverse_sqrt_values)
      s_permuted = Matrix.send(:new, w) * q * d_inverse
      s_rows = Array.new(size) { Array.new(size, 0.0) }
      size.times do |i|
        size.times do |column|
          s_rows[permutation[i]][column] = s_permuted[i, column]
        end
      end

      @symplectic_transform = Matrix.send(:new, fix_signs(s_rows))
      @j = Matrix.send(:new, j)
      @d = Matrix.diagonal(*normalized_eigenvalues.map { |value| value * scale })
    end

    def symplectic_eigenvalue_matrix
      @d
    end
    alias_method :d, :symplectic_eigenvalue_matrix
    alias_method :positive_eigenvalues, :symplectic_eigenvalues

    def canonical_symplectic_matrix
      @j
    end
    alias_method :j, :canonical_symplectic_matrix

    def to_ary
      [@symplectic_transform, @d, @symplectic_transform.transpose]
    end
    alias_method :to_a, :to_ary

    private

    def symmetric?(entries, scale, tolerance)
      size = entries.size
      size.times do |i|
        ((i + 1)...size).each do |column|
          difference = (entries[i][column] - entries[column][i]).to_f.abs
          return false if difference > tolerance * [scale, 1.0].max
        end
      end
      true
    end

    def pivoted_ldl(source)
      size = source.size
      a = source.map(&:dup)
      lower = Array.new(size) { Array.new(size, 0.0) }
      diagonal = Array.new(size, 0.0)
      permutation = (0...size).to_a

      size.times do |k|
        pivot = (k...size).max_by { |i| a[i][i].abs }
        if pivot != k
          a[k], a[pivot] = a[pivot], a[k]
          lower[k], lower[pivot] = lower[pivot], lower[k]
          size.times { |i| a[i][k], a[i][pivot] = a[i][pivot], a[i][k] }
          permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
        end

        pivot_value = a[k][k]
        if pivot_value <= 0.0
          raise ErrNotPositiveDefinite, "Matrix is not positive definite within numerical precision"
        end

        diagonal[k] = pivot_value
        lower[k][k] = 1.0
        ((k + 1)...size).each do |i|
          lower[i][k] = a[i][k] / pivot_value
          ((k + 1)...size).each do |column|
            a[i][column] -= lower[i][k] * a[k][column]
          end
        end
        ((k + 1)...size).each do |i|
          ((k + 1)...i).each do |column|
            a[column][i] = a[i][column]
          end
        end
      end

      [lower, diagonal, permutation]
    end

    def canonical_j(pair_count)
      size = pair_count * 2
      Array.new(size) do |i|
        Array.new(size) do |column|
          if i.even? && column == i + 1
            1.0
          elsif i.odd? && column == i - 1
            -1.0
          else
            0.0
          end
        end
      end
    end

    def multiply_transpose_left(w, j)
      size = w.size
      jw = Array.new(size) do |i|
        Array.new(size) do |column|
          sum = 0.0
          size.times { |r| sum += j[i][r] * w[r][column] }
          sum
        end
      end
      Array.new(size) do |i|
        Array.new(size) do |column|
          sum = 0.0
          size.times { |r| sum += w[r][i] * jw[r][column] }
          sum
        end
      end
    end

    def skew_part(matrix)
      size = matrix.size
      Array.new(size) do |i|
        Array.new(size) do |column|
          i == column ? 0.0 : (matrix[i][column] - matrix[column][i]) / 2.0
        end
      end
    end

    def skew_singular_vectors(k, tolerance)
      size = k.size
      vectors = Array.new(size) { Array.new(size, 0.0) }
      size.times { |i| vectors[i][i] = 1.0 }
      transformed = k.map(&:dup)
      max_sweeps = [30, 5 * size].max

      max_sweeps.times do
        off_norm = 0.0
        diagonal_norm = 0.0
        size.times do |p|
          ((p + 1)...size).each do |q|
            alpha = dot_column(transformed, p, p)
            beta = dot_column(transformed, q, q)
            gamma = dot_column(transformed, p, q)
            off_norm += gamma * gamma
            diagonal_norm += alpha * beta
            next if gamma.abs <= Float::EPSILON * Math.sqrt([alpha * beta, Float::EPSILON].max)

            zeta = (beta - alpha) / (2.0 * gamma)
            tau = (zeta >= 0.0 ? 1.0 : -1.0) / (zeta.abs + Math.hypot(zeta, 1.0))
            c = 1.0 / Math.hypot(1.0, tau)
            s = tau * c
            size.times do |r|
              k_p = transformed[r][p]
              k_q = transformed[r][q]
              transformed[r][p] = c * k_p - s * k_q
              transformed[r][q] = s * k_p + c * k_q

              v_p = vectors[r][p]
              v_q = vectors[r][q]
              vectors[r][p] = c * v_p + s * v_q
              vectors[r][q] = -s * v_p + c * v_q
            end
          end
        end
        return paired_singular_vectors(k, vectors, tolerance) if off_norm <= tolerance * tolerance * [diagonal_norm, Float::EPSILON].max
      end

      raise RuntimeError, "Williamson decomposition did not converge"
    end

    def dot_column(vectors, p, q)
      sum = 0.0
      vectors.size.times { |r| sum += vectors[r][p] * vectors[r][q] }
      sum
    end

    def paired_singular_vectors(k, vectors, tolerance)
      size = k.size
      columns = vectors.transpose.map { |column| scale_vector(column, 1.0 / norm(column)) }
      values = columns.map { |column| norm(matrix_vector(k, column)) }
      order = (0...size).sort_by { |i| values[i] }
      ordered_values = order.map { |i| values[i] }
      ordered_vectors = order.map { |i| columns[i] }

      pairs = []
      eigenvalues = []
      start = 0
      while start < size
        finish = start + 1
        while finish < size
          previous = ordered_values[finish - 1]
          current = ordered_values[finish]
          break unless (current - previous).abs <= tolerance * [current, 1.0].max
          finish += 1
        end

        cluster = ordered_vectors[start...finish].map(&:dup)
        cluster_value = ordered_values[start...finish].inject(:+) / (finish - start)
        raise RuntimeError, "Williamson decomposition produced an invalid singular subspace" unless cluster.size.even?

        until cluster.empty?
          q = remove_best_vector(cluster, pairs)
          p = scale_vector(matrix_vector(k, q), -1.0 / cluster_value)
          project_out!(p, pairs)
          p_norm = norm(p)
          raise RuntimeError, "Williamson decomposition produced an invalid symplectic pair" if p_norm < 0.5
          p = scale_vector(p, 1.0 / p_norm)
          remove_aligned_vector!(cluster, p)
          pairs << q << p
          eigenvalues << cluster_value << cluster_value
        end
        start = finish
      end

      [pairs, eigenvalues]
    end

    def remove_best_vector(vectors, basis)
      candidates = vectors.map do |vector|
        candidate = vector.dup
        project_out!(candidate, basis)
        [norm(candidate), candidate]
      end
      index = (0...candidates.size).max_by { |i| candidates[i][0] }
      _, best = candidates[index]
      vectors.delete_at(index)
      scale_vector(best, 1.0 / norm(best))
    end

    def remove_aligned_vector!(vectors, target)
      index = (0...vectors.size).max_by do |i|
        vectors[i].each_with_index.inject(0.0) { |sum, (value, r)| sum + value * target[r] }.abs
      end
      vectors.delete_at(index || 0)
    end

    def project_out!(vector, basis)
      basis.each do |direction|
        coefficient = 0.0
        vector.each_with_index { |value, i| coefficient += value * direction[i] }
        vector.each_with_index { |_, i| vector[i] -= coefficient * direction[i] }
      end
      vector
    end

    def matrix_vector(matrix, vector)
      matrix.map do |row|
        sum = 0.0
        row.each_with_index { |value, i| sum += value * vector[i] }
        sum
      end
    end

    def scale_vector(vector, factor)
      vector.map { |value| value * factor }
    end

    def norm(vector)
      Math.sqrt(vector.inject(0.0) { |sum, value| sum + value * value })
    end

    def fix_signs(rows)
      pair_count = rows.size / 2
      pair_count.times do |pair|
        q_column = pair * 2
        p_column = q_column + 1
        row = column = nil
        best = 0.0
        rows.each_with_index do |_, r|
          [q_column, p_column].each do |c|
            value = rows[r][c].abs
            if value > best
              best = value
              row = r
              column = c
            end
          end
        end
        if rows[row][column] < 0
          rows.each { |r| r[q_column] = -r[q_column]; r[p_column] = -r[p_column] }
        end
      end
      rows
    end
  end
end
