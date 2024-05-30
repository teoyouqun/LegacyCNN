#include <iostream>
#include "layers/conv1d.h"
#include "layers/dense.h"
#include "layers/helper.h"
#include "layers/basiccnnblock.h"
#include "models/basiccnn.h"

int main()
{
    // This layer is statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    // Refer to Documentation to generate the values using python.

    // Kernal, Stride, Channel_in, Channel_out, pad, dilation, input_width, out_dim

    // Conv1d<3, 1, 2, 3, 0, 2, 6, 2> layer1;
    // float weights[3][2][3] = {{{0, 0, 0}, {0, 0, 0}},
    //                           {{1, 1, 1}, {1, 1, 1}},
    //                           {{2, 2, 2}, {2, 2, 2}}};
    // float bias[3] = {0, 1, 2};

    // layer1.setWeights(weights);
    // layer1.setBias(bias);

    // float input[2][6] = {{1, 2, 3, 1, 2, 3}, {4, 5, 6, 4, 5, 6}};
    // float output[3][2];
    // layer1.getOutput(input, output);

    // Helper::Softmax(output);

    // float flatten_output[3 * 2];

    // Helper::Flatten(output, flatten_output);
    // for (int i = 0; i < 6; i++)
    // {
    //     std::cout << flatten_output[i] << " ";
    // }
    // std::cout << std::endl;

    // // FC Layer
    // Dense<6, 2> FClayer;
    // float FC_weights[6][2] = {{0, 1},
    //                           {0, 1},
    //                           {0, 1},
    //                           {0, 1},
    //                           {0, 1},
    //                           {0, 1}};
    // float FC_bias[2] = {2, 5};
    // FClayer.setWeights(FC_weights);
    // FClayer.setBias(FC_bias);

    // float y[2];
    // FClayer.getOutput(flatten_output, y);

    // for (int i = 0; i < 2; i++)
    // {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;

    // BasicCNNBlock<3, 1, 2, 4, 0, 1, 16, 14> Block0;
    // float input2[2][16] = {{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
    //                        {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7}};
    // float x0[4][14];

    // Block0.getOutput(input2, x0);

    // for (int c = 0; c < 4; c++)
    // {
    //     for (int i = 0; i < 14; i++)
    //     {
    //         std::cout << x0[c][i] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Initialise Model" << std::endl;
    // BasicCNNModel Model;
    // // std::cout << "Complete" << std::endl;
    // float y2[6];
    // std::cout << "Running Model" << std::endl;
    // // Model.forward(input2, y2);
    // std::cout << "Completed Running Model" << std::endl;

    float input2[2][16] = {{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
                           {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7}};
    float y[6];

    BasicCNNBlock<3, 1, 2, 4, 0, 1, 16, 14> Block0;
    Block0.setBias_layer0("BasicModelWeights\\layer0_conv_bias.bin");
    Block0.setWeights_layer0("BasicModelWeights\\layer0_conv_weights.bin");
    Block0.setGamma_layer1("BasicModelWeights\\layer0_bn_weights.bin");
    Block0.setBeta_layer1("BasicModelWeights\\layer0_bn_bias.bin");
    float x0[4][14];
    BasicCNNBlock<3, 1, 4, 4, 0, 1, 14, 12> Block1;
    Block1.setBias_layer0("BasicModelWeights\\layer1_conv_bias.bin");
    Block1.setWeights_layer0("BasicModelWeights\\layer1_conv_weights.bin");
    Block1.setGamma_layer1("BasicModelWeights\\layer0_bn_weights.bin");
    Block1.setBeta_layer1("BasicModelWeights\\layer0_bn_bias.bin");
    float x1[4][12];
    BasicCNNBlock<3, 1, 4, 4, 0, 1, 12, 10> Block2;
    Block2.setBias_layer0("BasicModelWeights\\layer2_conv_bias.bin");
    Block2.setWeights_layer0("BasicModelWeights\\layer2_conv_weights.bin");
    Block2.setGamma_layer1("BasicModelWeights\\layer0_bn_weights.bin");
    Block2.setBeta_layer1("BasicModelWeights\\layer0_bn_bias.bin");
    float x2[4][10];
    BasicCNNBlock<3, 1, 4, 4, 0, 1, 10, 8> Block3;
    Block3.setBias_layer0("BasicModelWeights\\layer3_conv_bias.bin");
    Block3.setWeights_layer0("BasicModelWeights\\layer3_conv_weights.bin");
    Block3.setGamma_layer1("BasicModelWeights\\layer0_bn_weights.bin");
    Block3.setBeta_layer1("BasicModelWeights\\layer0_bn_bias.bin");
    float x3[4][8];
    BasicCNNBlock<3, 1, 4, 4, 0, 1, 8, 6> Block4;
    Block4.setBias_layer0("BasicModelWeights\\layer4_conv_bias.bin");
    Block4.setWeights_layer0("BasicModelWeights\\layer4_conv_weights.bin");
    Block4.setGamma_layer1("BasicModelWeights\\layer0_bn_weights.bin");
    Block4.setBeta_layer1("BasicModelWeights\\layer0_bn_bias.bin");
    float x4[4][6];
    // Flatten()
    float flatten_x[24];
    Dense<24, 6> fc;
    fc.setBias("BasicModelWeights\\final_fc_bias.bin");
    fc.setWeights("BasicModelWeights\\final_fc_weights.bin");

    Block0.getOutput(input2, x0);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 14; j++)
        {
            std::cout << x0[i][j] << " ";
        }
        std::cout << std::endl;
    }

    Block1.getOutput(x0, x1);
    Block2.getOutput(x1, x2);

    Block3.getOutput(x2, x3);
    Block4.getOutput(x3, x4);
    Helper::Flatten(x4, flatten_x);
    fc.getOutput(flatten_x, y);
    Helper::Softmax(y);

    for (int i = 0; i < 6; i++)
    {
        std::cout << y[i] << " ";
    }
    return 0;
}