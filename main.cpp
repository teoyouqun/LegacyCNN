#include <iostream>
#include "tests/time.h"
#include "utils/matrixfunctions.h"
#include "utils/helper.h"
#include "models/ecapa_classifier.h"
#include "layers/seblock.h"
#include "layers/seres2netblock.h"

int main()
{
    /*
    // Time::TestEcapa(100);
    // Time::TestBasic(100);
    // Time::TestEcapaClassifier(100);

    std::cout << "Testing Ecapa" << std::endl;

    // Initialise model

    // Cosine = 0, CDist = 1, Euclidean = 2
    ECAPA_TDNN_classifier ecapamodel(0);
    float y_01[6][6]; // Used for Cosine/Cdist

    // Load weights
    ecapamodel.loadweights("ECAPAweights/fullecapa_classifier.bin");

    // Metrics on Read Input will not be tested
    float input[2][64];
    Helper::readInputs("ECAPAweights/ecapainput_2x64.bin", input);

    float lengths = 0.4;

    ecapamodel.forward(input, lengths, y_01);
    Helper::print(y_01);

    // We implement Euclidean version here.
    ECAPA_TDNN_classifier ecapamodel(0);
    float y_2[6]; // Used for Euclidean

    // Load weights
    ecapamodel.loadweights("ECAPAweights/fullecapa_classifier.bin");

    // Metrics on Read Input will not be tested
    float input[2][64];
    Helper::readInputs("ECAPAweights/ecapainput_2x64.bin", input);

    float lengths = 0.4;

    ecapamodel.forward(input, lengths, y_2);
    Helper::print(y_2);
    */
    //YQ
    std::cout << "Testing Ecapa" << std::endl;

    // Initialise model
    // ECAPA_TDNN ecapamodel;
    BasicCNNModel basicmodel;
    float y[6];

    // Load weights
    // ecapamodel.loadweights("verifyCppVsPy/fullecapa.bin");
    basicmodel.loadweights("verifyCppVsPy/fullbasicmodel.bin");

    // Metrics on Read Input will not be tested
    float input_full[10][2][16];
    float input[2][16];
    std::cout << "Before Loading" << std::endl;
    // Helper::readInputs("verifyCppVsPy/testInput_ecapaModel_100x2x64.bin", input_full);
    Helper::readInputs("verifyCppVsPy/testInput_basicModel_10x2x16.bin", input_full);
    std::cout << "Complete Loading" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        MatrixFunctions::Copy(input_full[i], input);
        // ecapamodel.forward(input, y);
        basicmodel.forward(input, y);
        Helper::print(y);
    }
    // YQ
    return 0;
}
