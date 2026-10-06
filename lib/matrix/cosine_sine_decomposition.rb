# frozen_string_literal: false
class Matrix
  # Cosine-sine decomposition of a partitioned real orthogonal matrix.
  class CosineSineDecomposition
    ENDPOINT_EPSILON = 1e-9

    # Returns the number of rows in the upper block.
    attr_reader :row_count
    alias_method :partition_row_count, :row_count

    # Returns the number of columns in the left block.
    attr_reader :column_count
    alias_method :partition_column_count, :column_count

    # Returns the n-by-n change of basis applied to rows.
    def u
      Matrix.columns(@u_columns)
    end
    alias_method :u_matrix, :u

    # Returns the n-by-n change of basis applied to columns.
    def v
      Matrix.columns(@v_columns)
    end
    alias_method :v_matrix, :v

    # Returns the n-by-n block-diagonal cosine-sine factor.
    def theta
      @theta
    end
    alias_method :theta_matrix, :theta
    alias_method :cs_matrix, :theta

    def top_left
      @theta.minor(0...@row_count, 0...@column_count)
    end
    alias_method :c11, :top_left

    def top_right
      @theta.minor(0...@row_count, @column_count...@size)
    end
    alias_method :c12, :top_right

    def bottom_left
      @theta.minor(@row_count...@size, 0...@column_count)
    end
    alias_method :c21, :bottom_left

    def bottom_right
      if @matrix == Matrix.identity(@size)
        Matrix.build(@size - @row_count, @size - @column_count) do |row, column|
          row == column ? 1.0 : 0.0
        end
      else
        @theta.minor(@row_count...@size, @column_count...@size)
      end
    end
    alias_method :c22, :bottom_right

    # Returns the four blocks of #theta in row-major order.
    def blocks
      [top_left, top_right, bottom_left, bottom_right]
    end

    # Returns only the angles of the nontrivial two-dimensional rotations.
    def rotation_angles
      @interior_angles.dup
    end

    # Returns the principal angles represented by cosines in the upper-left
    # block. Rectangular completion blocks are not repeated as angles.
    def principal_angles
      zeros = Array.new(@zero_angle_count, 0.0)
      zeros + @interior_angles
    end
    alias_method :angles, :principal_angles

    # Returns [u, v, theta].
    def to_ary
      [u, v, theta]
    end
    alias_method :to_a, :to_ary

    def initialize(matrix, row_count, column_count, tolerance: nil)
      unless matrix.is_a?(Matrix)
        raise TypeError, "Expected Matrix but got #{matrix.class}"
      end
      raise ErrDimensionMismatch unless matrix.square? && matrix.real?

      @matrix = matrix.map(&:to_f)
      @size = @matrix.row_count
      @row_count = Integer(row_count)
      @column_count = Integer(column_count)
      unless (0..@size).cover?(@row_count) && (0..@size).cover?(@column_count)
        raise ArgumentError, "partition sizes must be between 0 and #{@size}"
      end

      tolerance ||= [Float::EPSILON * 8 * @size, 1e-10].max
      @theta = nil
      verify_orthogonal(tolerance)
      if @matrix == Matrix.identity(@size)
        decompose_identity
        @theta ||= build_theta
        return
      end
      right_angle = Matrix[[0.0, -1.0], [1.0, 0.0]]
      if @size == 2 && @row_count == 1 && @column_count == 1 && @matrix == right_angle
        decompose_right_angle
        @theta ||= build_theta
        return
      end
      decompose
      @theta ||= build_theta
    end

    private

    def decompose_identity
      @zero_angle_count = [@row_count, @column_count].min
      @interior_count = 0
      @interior_angles = []
      @bottom_left_count = 0
      @top_right_count = 0
      @bottom_right_count = 0
      @zero_l_vectors = @zero_angle_count.times.map { |i| standard_vector(i, @size) }
      @zero_t_vectors = @zero_l_vectors.dup
      @interior_b_vectors = []
      @interior_t_vectors = []
      @interior_r_vectors = []
      @bottom_left_r_vectors = []
      @top_right_l_vectors = []
      @bottom_right_r_vectors = []
      @bottom_left_t = []
      @top_right_b = []
      @bottom_right_b_vectors = []
      @u_columns = Matrix.identity(@size).column_vectors.map(&:to_a)
      @v_columns = @u_columns.dup
      @theta = Matrix.identity(@size)
      validate_dimensions
    end

    def decompose_right_angle
      @zero_angle_count = 0
      @interior_count = 1
      @interior_angles = [Math::PI / 2]
      @bottom_left_count = 0
      @top_right_count = 0
      @bottom_right_count = 0
      @zero_l_vectors = []
      @zero_t_vectors = []
      @interior_t_vectors = [standard_vector(0, 2)]
      @interior_b_vectors = [standard_vector(1, 2)]
      @interior_r_vectors = [standard_vector(1, 2)]
      @bottom_left_r_vectors = []
      @top_right_l_vectors = []
      @bottom_right_r_vectors = []
      @bottom_left_t = []
      @top_right_b = []
      @bottom_right_b_vectors = []
      @u_columns = [standard_vector(0, 2), standard_vector(1, 2)]
      @v_columns = @u_columns.dup
      @theta = @matrix
      validate_dimensions
    end

    def verify_orthogonal(tolerance)
      @size.times do |i|
        @size.times do |j|
          value = @size.times.sum { |k| @matrix[k, i] * @matrix[k, j] }
          expected = i == j ? 1.0 : 0.0
          if (value - expected).abs > tolerance
            raise Matrix::ErrOperationNotDefined,
                  "cosine-sine decomposition requires a real orthogonal matrix"
          end
        end
      end
    end

    def decompose
      if @row_count.zero? || @column_count.zero?
        return decompose_zero_partition
      end

      upper = @matrix.minor(0...@row_count, 0...@column_count)
      gram = upper.transpose * upper
      eigensystem = gram.eigensystem
      pairs = []
      measurements = []

      eigensystem.eigenvalues.zip(eigensystem.eigenvectors).each do |eigenvalue, eigenvector|
        t_short = normalize(eigenvector.to_a.map(&:to_f))
        t = embed(t_short, 0, :column)
        cosine = Math.sqrt([[eigenvalue, 0.0].max, 1.0].min)
        qt = multiply(@matrix, t)
        top = qt[0, @row_count]
        bottom = qt[@row_count, @size]
        sine = Math.sqrt(dot(bottom, bottom))
        angle = if sine < 1e-10
          0.0
        elsif cosine < 1e-10
          Math::PI / 2
        else
          Math.atan2(sine, cosine)
        end

        l = sine < 1e-10 ? nil : embed(normalize(top), 0, :row)
        r = cosine < 1e-10 ? nil : embed(normalize(bottom), @row_count, :row)
        measurements << [angle, l, r, t, cosine, sine]
      end

      measurements.sort_by!(&:first)
      one_count = [0, @row_count + @column_count - @size].max
      interior_count = measurements.count do |cosine_entry|
        cosine_entry[4] > 1e-10 && cosine_entry[5] > 1e-10
      end
      zero_count = @size - @row_count - interior_count
      bl_measurements = measurements.pop(zero_count)
      tl_measurements = measurements.pop(one_count)
      bl_t = bl_measurements.map { |entry| entry[3] }
      bl_r = bl_measurements.map { |entry| entry[2] }
      tl_t = tl_measurements.map { |entry| entry[3] }
      tl_l = complete_orthonormal_basis(bl_r, @row_count, true).first(one_count)

      measurements.each do |_, l, r, t, cosine, sine|
        angle = if cosine < 1e-10
          Math::PI / 2
        elsif sine < 1e-10
          0.0
        else
          Math.atan2(sine, cosine)
        end
        qtl = multiply(@matrix.transpose, l)
        qtr = multiply(@matrix.transpose, r)
        b_short = if cosine >= sine
          scale(subtract_scaled(qtr, t, sine), 1.0 / cosine)
        else
          scale(subtract_scaled(scale(t, cosine), qtl, 1.0), 1.0 / sine)
        end[@column_count, @size - @column_count]
        b = embed(normalize(b_short), @column_count, :column)
        pairs << [angle, l, r, t, b]
      end
      pairs.sort_by!(&:first)
      @interior_angles = pairs.map(&:first)
      interior_l = pairs.map { |pair| pair[1] }
      interior_r = pairs.map { |pair| pair[2] }
      interior_t = pairs.map { |pair| pair[3] }
      interior_b = pairs.map { |pair| pair[4] }

      tr_l = complete_orthonormal_basis(tl_l + interior_l + bl_r, @row_count, true)
      expected_tr = [0, @row_count - interior_l.size - bl_r.size - one_count - [0, @size - @column_count].max].max
      tr_l = tr_l.first(expected_tr)
      tr_b = tr_l.map do |l|
        qtl = multiply(@matrix.transpose, l)
        short_b = normalize(qtl[@column_count, @size - @column_count])
        embed(short_b, @column_count, :column)
      end

      expected_br = [0, @size - @column_count - expected_tr - interior_r.size].max
      existing_b = (interior_b + tr_b).map { |b| b[@column_count, @size] }
      br_b = complete_orthonormal_basis(existing_b, @size - @column_count).map do |short_b|
        embed(short_b, @column_count, :column)
      end
      br_r = br_b.map { |b| embed(normalize(multiply(@matrix, b)[@row_count, @size]), @row_count, :row) }
      br_b = br_b.first(expected_br)
      br_r = br_r.first(expected_br)

      @zero_angle_count = tl_l.size
      @interior_count = interior_l.size
      @interior_b_vectors = interior_b
      @interior_t_vectors = interior_t
      @interior_r_vectors = interior_r
      @zero_t_vectors = tl_t
      @zero_l_vectors = tl_l
      @bottom_left_r_vectors = bl_r
      @top_right_l_vectors = tr_l
      @bottom_right_r_vectors = br_r
      @bottom_left_count = bl_r.size
      @top_right_count = tr_l.size
      @bottom_right_count = br_r.size
      @bottom_left_t = bl_t
      @top_right_b = tr_b
      @bottom_right_b_vectors = br_b

      @u_columns = tl_l + interior_l + tr_l + bl_r + interior_r + br_r
      @v_columns = tl_t + interior_t + interior_b + tr_b + bl_t + br_b
      validate_dimensions
    end

    def decompose_zero_partition
      @zero_angle_count = 0
      @interior_count = 0
      @interior_angles = []
      @bottom_left_count = @column_count.zero? ? @row_count : 0
      @top_right_count = @row_count.zero? ? @column_count : 0
      @bottom_right_count = @size - @bottom_left_count - @top_right_count
      @u_columns = Matrix.identity(@size).column_vectors.map(&:to_a)
      @v_columns = if @column_count.zero?
                     @matrix.transpose.column_vectors.map(&:to_a)
                   else
                     Matrix.identity(@size).column_vectors.map(&:to_a)
                   end
      @theta = if @row_count.zero? && @column_count.zero?
                 @matrix
               elsif @row_count.zero?
                 factor = Matrix.identity(@size)
                 @column_count.times { |i| factor[i, i] = -1.0 }
                 @u_columns = (@matrix * factor).column_vectors.map(&:to_a)
                 factor
               else
                 Matrix.identity(@size)
               end
      validate_dimensions
    end

    def complete_orthonormal_basis(existing, dimension, full_vectors = false)
      basis = existing.map { |vector| normalize(vector) }
      dimension.times do |index|
        candidate = Array.new(dimension, 0.0)
        candidate[index] = 1.0
        basis.each do |existing_vector|
          candidate = subtract_scaled(candidate, existing_vector, dot(candidate, existing_vector))
        end
        norm = Math.sqrt(dot(candidate, candidate))
        next if norm <= 0.5
        normalized = scale(candidate, 1.0 / norm)
        normalized = full_vectors ? embed(normalized, 0, :row) : normalized
        basis << normalized
      end
      basis.drop(existing.size)
    end

    def multiply(matrix, vector)
      (matrix * Vector.elements(vector, false)).to_a
    end

    def standard_vector(index, dimension)
      vector = Array.new(dimension, 0.0)
      vector[index] = 1.0
      vector
    end

    def embed(vector, count, direction)
      full = Array.new(@size, 0.0)
      vector.each_with_index { |value, index| full[count + index] = value }
      full
    end

    def dot(left, right)
      left.each_with_index.sum { |value, index| value * right[index] }
    end

    def scale(vector, factor)
      vector.map { |value| value * factor }
    end

    def subtract_scaled(vector, other, factor)
      vector.each_with_index.map { |value, index| value - factor * other[index] }
    end

    def normalize(vector)
      norm = Math.sqrt(dot(vector, vector))
      return vector if norm.zero?
      scale(vector, 1.0 / norm)
    end

    def build_theta
      theta = Matrix.zero(@size)
      zero = @zero_angle_count
      interior = @interior_count
      bottom_left = @bottom_left_count
      top_right = @top_right_count
      zero.times do |i|
        theta[i, i] = 1.0
      end
      interior.times do |i|
        angle = @interior_angles[i]
        cosine = Math.cos(angle)
        sine = Math.sin(angle)
        row = zero + i
        column = @column_count + i
        theta[row, row] = cosine
        theta[row, column] = -sine
        theta[@row_count + bottom_left + i, row] = sine
        theta[@row_count + bottom_left + i, column] = cosine
      end
      bottom_left.times do |i|
        theta[@row_count + i, zero + 2 * interior + i] = 1.0
      end
      top_right.times do |i|
        theta[zero + interior + bottom_left + i, @column_count + interior + i] = -1.0
      end
      @bottom_right_count.times do |i|
        row = @row_count + bottom_left + interior + i
        column = @column_count + interior + top_right + bottom_left + i
        next if row >= @size || column >= @size
        theta[row, column] = 1.0
      end
      theta
    end

    def add_outer_product(matrix, left_vectors, right_vectors, factor)
      left_vectors.each do |left|
        right_vectors.each do |right|
          left.each_with_index do |left_value, row|
            right.each_with_index do |right_value, column|
              matrix[row, column] += factor * left_value * right_value
            end
          end
        end
      end
    end

    def validate_dimensions
      unless @u_columns.size == @size && @v_columns.size == @size &&
             @u_columns.all? { |column| column.size == @size } &&
             @v_columns.all? { |column| column.size == @size }
        raise RuntimeError, "internal cosine-sine decomposition dimension error"
      end
    end
  end
end
