#ifndef dense_h
#define dense_h

#include <iostream>
#include <assert.h>
#include <string>
#include <fstream>
template <int input_dim, int output_dim, typename T>
class Dense
{
public:
    Dense()
    {

        // Initialize the weights with 0
        for (int i = 0; i < output_dim; i++)
        {
            for (int j = 0; j < input_dim; j++)
            {
                this->weights[i][j] = 0.0;
            }
        }

        // Initialize the bias with 0
        for (int i = 0; i < output_dim; i++)
        {
            this->bias[i] = 0.0;
        }

        std::cout << "Dense initialised" << std::endl;
    };

    void setWeights(T (&new_weights)[input_dim][output_dim])
    {
        // New weights are being set
        for (int i = 0; i < input_dim; i++)
        {
            for (int j = 0; j < output_dim; j++)
            {
                // std::cout << i << j << " " << new_weights[i][j] << std::endl;
                this->weights[i][j] = new_weights[i][j];
            }
        }
    };

    void setBias(T (&new_bias)[output_dim])
    {
        // New bias are being set
        for (int i = 0; i < output_dim; i++)
        {
            this->bias[i] = new_bias[i];
        }
    };

    // Overloading
    void setBias(std::string filename)
    {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile)
        {
            std::cout << "Error opening file!" << std::endl;
            return;
        }
        // Read dimensions
        int dim1;
        infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));

        // Sanity Check
        assert(dim1 == output_dim);

        // Calculate total size
        int total_size = dim1;

        // Read flattened array
        infile.read(reinterpret_cast<char *>(this->bias), total_size * sizeof(T));
        infile.close();
    };

    // Overloading
    void setWeights(std::string filename, bool displayWeights)
    {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile)
        {
            std::cout << "Error opening file!" << std::endl;
            return;
        }
        // Read dimensions
        int dim1, dim2;
        infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));
        infile.read(reinterpret_cast<char *>(&dim2), sizeof(int));

        // Sanity Check
        assert(dim1 == output_dim);
        assert(dim2 == input_dim);

        // Calculate total size
        int total_size = dim1 * dim2;

        // Read flattened array
        infile.read(reinterpret_cast<char *>(this->flat_weights), total_size * sizeof(T));
        infile.close();

        for (int i = 0; i < dim1; ++i)
        {
            for (int j = 0; j < dim2; ++j)
            {
                this->weights[i][j] = (T)this->flat_weights[i * dim2 + j];
            }
        }

        if (displayWeights)
        {
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    std::cout << this->weights[i][j] << " ";
                }
                std::cout << std::endl;
                std::cout << std::endl;
            }
        }
    };

    void getOutput(T (&input)[input_dim], T (&output)[output_dim])
    {
        for (int i = 0; i < output_dim; i++)
        {
            output[i] = this->bias[i];
            for (int j = 0; j < input_dim; j++)
            {
                output[i] += this->weights[i][j] * input[j];
            }
        }
    }

private:
    T weights[output_dim][input_dim];
    T flat_weights[output_dim * input_dim];
    T bias[output_dim];
    float flat_bias[output_dim];
};

#endif