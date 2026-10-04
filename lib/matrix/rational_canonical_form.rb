# frozen_string_literal: false
class Matrix
  class RationalCanonicalForm
    # Returns the Frobenius companion blocks, the invertible change-of-basis
    # matrix +P+, and the monic invariant factors.
    attr_reader :frobenius_blocks, :change_of_basis_matrix, :invariant_factors

    alias_method :blocks, :frobenius_blocks
    alias_method :p, :change_of_basis_matrix

    # Constructs the rational canonical form of matrix +a+.
    def initialize(a)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch unless a.square?

      a.each(:all) do |entry|
        unless entry.is_a?(Integer) || entry.is_a?(Rational)
          raise TypeError, "rational canonical form requires Integer or Rational entries"
        end
      end

      blocks, @change_of_basis_matrix, @invariant_factors = rational_canonical_form(a)
      @frobenius_blocks = blocks
    end

    # Returns the block diagonal matrix formed from the Frobenius blocks.
    def frobenius_matrix
      RationalCanonicalForm.block_diagonal(@frobenius_blocks)
    end
    alias_method :f, :frobenius_matrix

    def to_ary
      [@frobenius_blocks, @change_of_basis_matrix, @invariant_factors]
    end
    alias_method :to_a, :to_ary

    def self.block_diagonal(blocks)
      size = blocks.sum(&:column_count)
      rows = Array.new(size) { Array.new(size, 0) }
      row_offset = 0

      blocks.each do |block|
        block.to_a.each_with_index do |block_row, row|
          block_row.each_with_index do |value, column|
            rows[row_offset + row][row_offset + column] = value
          end
        end
        row_offset += block.column_count
      end

      Matrix.send(:new, rows, size)
    end

    private

    def rational_canonical_form(a)
      return [[], Matrix.empty, []] if a.row_count.zero?

      cyclic_vector, minimal_polynomial = cyclic_vector_with_minimal_polynomial(a)
      degree = polynomial_degree(minimal_polynomial)

      cyclic_vectors = Array.new(degree) do |power|
        matrix_power_times_vector(a, power, cyclic_vector)
      end
      cyclic_basis = Matrix.columns(cyclic_vectors.map(&:to_a))
      completion = complete_basis(cyclic_basis)
      full_basis = Matrix.hstack(cyclic_basis, completion)
      representation = full_basis.lup.solve(a * full_basis)

      top = degree
      size = a.row_count
      d = representation.minor(0...top, 0...top)
      c = representation.minor(0...top, top...size)
      e = representation.minor(top...size, top...size)

      y = solve_sylvester(d, e, -c)
      invariant_completion = completion + cyclic_basis * y

      sub_blocks, sub_basis, sub_factors = rational_canonical_form(e)
      lifted_sub_basis = invariant_completion * sub_basis
      basis = Matrix.hstack(lifted_sub_basis, cyclic_basis)

      blocks = sub_blocks + [companion_matrix(minimal_polynomial)]
      factors = sub_factors + [minimal_polynomial]
      [blocks, basis, factors]
    end

    def cyclic_vector_with_minimal_polynomial(a)
      size = a.row_count
      minimal_polynomial = [1]

      size.times do |index|
        vector = Vector.elements(Array.new(size) { |column| column == index ? 1 : 0 })
        minimal_polynomial = polynomial_lcm(
          minimal_polynomial,
          vector_minimal_polynomial(a, vector)
        )
      end

      parameter = 0
      loop do
        vector = Vector.elements(Array.new(size) { |power| parameter ** power })
        vector_polynomial = vector_minimal_polynomial(a, vector)
        return vector, vector_polynomial if
          polynomial_degree(vector_polynomial) == polynomial_degree(minimal_polynomial)

        parameter += 1
      end
    end

    def complete_basis(basis)
      size = basis.row_count
      columns = basis.column_vectors
      return Matrix.empty(size, 0) if columns.size == size

      size.times do |index|
        vector = Vector.elements(Array.new(size) { |column| column == index ? 1 : 0 })
        candidate_columns = columns + [vector]
        candidate = Matrix.columns(candidate_columns.map(&:to_a))
        columns << vector if candidate.rank == candidate_columns.size
      end

      Matrix.columns(columns[basis.column_count..-1].map(&:to_a))
    end

    def solve_sylvester(d, e, right_hand_side)
      cyclic_dimension = d.column_count
      complement_dimension = e.row_count
      variables = cyclic_dimension * complement_dimension
      return Matrix.empty(cyclic_dimension, 0) if variables.zero?

      equations = Array.new(variables) { Array.new(variables, 0) }
      constants = Array.new(variables, 0)

      cyclic_dimension.times do |row|
        complement_dimension.times do |column|
          equation = row * complement_dimension + column
          constants[equation] = right_hand_side[row, column]

          cyclic_dimension.times do |variable_row|
            complement_dimension.times do |variable_column|
              variable = variable_row * complement_dimension + variable_column
              coefficient = 0
              coefficient += d[row, variable_row] if variable_column == column
              coefficient -= e[variable_column, column] if variable_row == row
              equations[equation][variable] = coefficient
            end
          end
        end
      end

      solution = solve_linear_system(Matrix.rows(equations, false), Vector.elements(constants, false))
      values = Array.new(cyclic_dimension) do |row|
        Array.new(complement_dimension) do |column|
          solution[row * complement_dimension + column]
        end
      end
      Matrix.rows(values, false)
    end

    def vector_minimal_polynomial(a, vector)
      vectors = [vector]
      current = a * vector

      1.upto(a.row_count) do |degree|
        coefficients = solve_linear_system(Matrix.columns(vectors.map(&:to_a)), -current)
        if coefficients
          polynomial = Array.new(degree + 1) { |power| coefficients[power] || 0 }
          polynomial[degree] = 1
          return normalize_polynomial(polynomial)
        end

        vectors << current
        current = a * current
      end
    end

    def solve_linear_system(matrix, right_hand_side)
      row_count = matrix.row_count
      columns = matrix.column_count
      right_hand_columns = right_hand_side.is_a?(Vector) ? 1 : right_hand_side.column_count
      augmented = Array.new(row_count) do |row|
        Array.new(columns) { |column| matrix[row, column].quo(1) } +
          Array.new(right_hand_columns) do |column|
            value = right_hand_side.is_a?(Vector) ? right_hand_side[row] : right_hand_side[row, column]
            value.quo(1)
          end
      end

      pivot_row = 0
      pivot_columns = []

      columns.times do |pivot_column|
        next unless pivot_row < row_count

        selected = (pivot_row...row_count).find { |row| augmented[row][pivot_column] != 0 }
        next unless selected

        augmented[pivot_row], augmented[selected] = augmented[selected], augmented[pivot_row]
        pivot = augmented[pivot_row][pivot_column]

        (pivot_column...columns + right_hand_columns).each do |column|
          augmented[pivot_row][column] /= pivot
        end

        row_count.times do |row|
          next if row == pivot_row
          factor = augmented[row][pivot_column]
          next if factor == 0

          (pivot_column...columns + right_hand_columns).each do |column|
            augmented[row][column] -= factor * augmented[pivot_row][column]
          end
        end

        pivot_columns[pivot_row] = pivot_column
        pivot_row += 1
      end

      row_count.times do |row|
        if (0...columns).all? { |column| augmented[row][column] == 0 } &&
           (columns...columns + right_hand_columns).any? { |column| augmented[row][column] != 0 }
          return nil
        end
      end

      values = Array.new(right_hand_columns) { Array.new(columns, 0) }
      pivot_row.times do |row|
        pivot_column = pivot_columns[row]
        right_hand_columns.times do |right_column|
          values[right_column][pivot_column] = augmented[row][columns + right_column]
        end
      end

      if right_hand_side.is_a?(Vector)
        Vector.elements(values.first, false)
      else
        Matrix.rows(values.first(right_hand_columns).map { |row| row }, false).transpose
      end
    end

    def matrix_power_times_vector(a, power, vector)
      result = vector
      power.times { result = a * result }
      result
    end

    def companion_matrix(polynomial)
      degree = polynomial_degree(polynomial)
      rows = Array.new(degree) { Array.new(degree, 0) }

      (degree - 1).times do |index|
        rows[index + 1][index] = 1
      end
      degree.times do |index|
        rows[index][degree - 1] = -polynomial[index]
      end

      Matrix.rows(rows, false)
    end

    def polynomial_degree(polynomial)
      polynomial.size - 1
    end

    def normalize_polynomial(polynomial)
      polynomial = polynomial.dup
      polynomial.pop while polynomial.size > 1 && polynomial.last.zero?
      return polynomial if polynomial == [0]
      return polynomial if polynomial.last == 1

      leading_coefficient = polynomial.last
      polynomial.map { |coefficient| coefficient / leading_coefficient }
    end

    def polynomial_divmod(dividend, divisor)
      divisor = normalize_polynomial(divisor)
      quotient_degree = polynomial_degree(dividend) - polynomial_degree(divisor)
      return [[0], dividend] if quotient_degree < 0

      quotient = Array.new(quotient_degree + 1, 0)
      remainder = dividend.dup

      quotient_degree.downto(0) do |degree|
        factor = remainder[degree + polynomial_degree(divisor)] / divisor.last
        quotient[degree] = factor
        (0..polynomial_degree(divisor)).each do |power|
          remainder[degree + power] -= factor * divisor[power]
        end
      end

      [quotient, remainder]
    end

    def polynomial_gcd(left, right)
      left = normalize_polynomial(left)
      right = normalize_polynomial(right)

      while right != [0]
        _, remainder = polynomial_divmod(left, right)
        left = right
        right = normalize_polynomial(remainder)
      end

      normalize_polynomial(left)
    end

    def polynomial_lcm(left, right)
      gcd = polynomial_gcd(left, right)
      quotient, _ = polynomial_divmod(left, gcd)
      normalize_polynomial(polynomial_multiply(quotient, right))
    end

    def polynomial_multiply(left, right)
      product = Array.new(polynomial_degree(left) + polynomial_degree(right) + 1, 0)
      left.each_with_index do |left_coefficient, left_power|
        right.each_with_index do |right_coefficient, right_power|
          product[left_power + right_power] += left_coefficient * right_coefficient
        end
      end
      product
    end
  end
end
