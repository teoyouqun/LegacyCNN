#ifndef matrixfunctions_h
#define matrixfunctions_h

#include <string>
#include <cmath>
#include <iostream>
#include <assert.h>

class MatrixFunctions
{
public:
    template <size_t rows, size_t cols, typename T>
    static T Sum(T (&input)[rows][cols])
    {
        T total = 0.0f;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                total += input[i][j];
            }
        }
        return total;
    };

    template <size_t rows, typename T>
    static T Sum(T (&input)[rows])
    {
        T total = 0.0;
        for (int i = 0; i < rows; i++)
        {
            total += input[i];
        }
        return total;
    };

    template <size_t rows, size_t cols, typename T>
    static void Flatten(T (&input)[rows][cols], T (&output)[rows * cols])
    {
        int index = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                output[index++] = input[i][j];
            }
        }
    };
    // mat1 += mat2
    // The output will be on mat1
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void matrixAdd(T (&mat1)[dim1][dim2][dim3], T (&mat2)[dim1][dim2][dim3])
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    mat1[i][j][k] += mat2[i][j][k];
                }
            }
        }
    };

    // mat1 += mat2
    // The output will be on mat1
    template <size_t dim1, size_t dim2, typename T>
    static void matrixAdd(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2])
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                mat1[i][j] += mat2[i][j];
            }
        }
    };

    // mat1 += mat2
    // The output will be on mat1
    template <size_t dim1, typename T>
    static void matrixAdd(T (&mat1)[dim1], T (&mat2)[dim1])
    {
        for (int i = 0; i < dim1; i++)
        {
            mat1[i] += mat2[i];
        }
    };

    //
    template <size_t dim1, size_t dim2, size_t chunk, typename T>
    static void Chunk(T (&input)[chunk * dim1][dim2], T (&output)[chunk][dim1][dim2])
    {
        for (int i = 0; i < chunk; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    output[i][j][k] = input[i * dim1 + j][k];
                    // std::cout << i << j << " " << i * dim1 + j << " " << output[i][j][k] << std::endl;
                }
            }
        }
    };

    template <size_t dim1, size_t dim2, size_t chunk, typename T>
    static void Cat(T (&input)[chunk][dim1][dim2], T (&output)[chunk * dim1][dim2])
    {
        for (int i = 0; i < chunk; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    output[i * dim1 + j][k] = input[i][j][k];
                }
            }
        }
    };

    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void Copy(T (&input)[dim1][dim2][dim3], T (&output)[dim1][dim2][dim3])
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    output[i][j][k] = input[i][j][k];
                }
            }
        }
    };
    template <size_t dim1, size_t dim2, typename T>
    static void Copy(T (&input)[dim1][dim2], T (&output)[dim1][dim2])
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                output[i][j] = input[i][j];
            }
        }
    };
    template <size_t dim1, typename T>
    static void Copy(T (&input)[dim1], T (&output)[dim1])
    {
        for (int i = 0; i < dim1; i++)
        {
            output[i] = input[i];
        }
    };

    template <size_t dim1, size_t dim2, typename T>
    static void HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2], T (&output)[dim1][dim2])
    {
        assert(dim2 == dim3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                output[i][j] = mat1[i][j] * mat2[i][j];
            }
        }
    }

    // Hadamard Product here aim to replicate the Broadcasting feature in PyTorch
    template <size_t dim1, size_t dim2, typename T>
    static void HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1], T (&output)[dim1][dim2])
    {
        assert(dim2 == dim3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                output[i][j] = mat1[i][j] * mat2[i];
            }
        }
    }

    // Mean will be based on dim2
    template <size_t dim1, size_t dim2, typename T>
    static void Mean(T (&input)[dim1][dim2], T (&output)[dim1])
    {
        for (int i = 0; i < dim1; i++)
        {
            output = Sum(input[i]) / dim2;
        }
    }
};

#endif