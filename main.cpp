#include <iostream>
#include <cstdlib>
#include "tests/time.h"
#include "utils/matrixfunctions.h"
#include "utils/helper.h"
#include "models/ecapa_classifier.h"
#include "layers/seblock.h"
#include "layers/seres2netblock.h"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> " << std::endl;
        std::cerr << "\tmode 0 = Basic Model" << std::endl;
        std::cerr << "\tmode 1 = ECAPA Model" << std::endl;
        std::cerr << "\tmode 2 = ECAPA Classifier - Cosine" << std::endl;
        std::cerr << "\tmode 3 = ECAPA Classifier - CDist" << std::endl;
        std::cerr << "\tmode 4 = ECAPA Classifier - Euclidean" << std::endl;
        return 1;
    }

    enum VerificationTest {
        basicModel = 0,
        ecapaModel = 1,
        ecapaClassifierCosine = 2,
        ecapaClassifierCdist = 3,
        ecapaClassifierEuclidean = 4
    };

    VerificationTest testMode = (VerificationTest)std::atoi(argv[1]);
    if (testMode < basicModel || testMode > ecapaClassifierEuclidean) {
        std::cerr << "Error: Invalid mode. Please provide a value between 0 and 4." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <mode> " << std::endl;
        std::cerr << "\tmode 0 = Basic Model" << std::endl;
        std::cerr << "\tmode 1 = ECAPA Model" << std::endl;
        std::cerr << "\tmode 2 = ECAPA Classifier - Cosine" << std::endl;
        std::cerr << "\tmode 3 = ECAPA Classifier - CDist" << std::endl;
        std::cerr << "\tmode 4 = ECAPA Classifier - Euclidean" << std::endl;
        return 1;
    }

    // Testing Basic Model
    if (testMode == basicModel) {
        std::cout << "Testing Basic Model" << std::endl;

        // Initialise model
        BasicCNNModel basicmodel;
        float y[6];

        // Load weights
        basicmodel.loadweights("verifyCppVsPy/fullbasicmodel.bin");

        // Metrics on Read Input will not be tested
        float input_full[10][2][16];
        float input[2][16];
        std::cout << "Before Loading" << std::endl;
        Helper::readInputs("verifyCppVsPy/testInput_basicModel_10x2x16.bin", input_full);
        std::cout << "Complete Loading" << std::endl;

        for (int i = 0; i < 10; i++)
        {
            MatrixFunctions::Copy(input_full[i], input);
            basicmodel.forward(input, y);
            Helper::print(y);
        }
    }

    // Testing Ecapa Model
    if (testMode == ecapaModel) {
        std::cout << "Testing Ecapa" << std::endl;

        // Initialise model
        ECAPA_TDNN ecapamodel;
        float y[6];

        // Load weights
        ecapamodel.loadweights("verifyCppVsPy/fullecapa.bin");

        // Metrics on Read Input will not be tested
        float input_full[100][2][64];
        float input[2][64];
        std::cout << "Before Loading" << std::endl;
        Helper::readInputs("verifyCppVsPy/testInput_ecapaModel_100x2x64.bin", input_full);
        std::cout << "Complete Loading" << std::endl;

        for (int i = 0; i < 100; i++)
        {
            MatrixFunctions::Copy(input_full[i], input);
            ecapamodel.forward(input, y);
            Helper::print(y);
        }
    }

    // Testing Ecapa Classifier COSINE
    if (testMode == ecapaClassifierCosine) {
        // Cosine = 0, CDist = 1, Euclidean = 2
        ECAPA_TDNN_classifier ecapamodel(0);
        float y_1[6][6]; // Used for Cosine/Cdist

        // Load weights
        ecapamodel.loadweights("verifyCppVsPy/fullecapa_classifier_cosine.bin");

        // Metrics on Read Input will not be tested
        float input_full[100][2][64];
        float input[2][64];
        Helper::readInputs("verifyCppVsPy/testInput_ecapaClassifier_cosine_100x2x64.bin", input_full);

        float lengths = 0.4;

        for (int i = 0; i < 100; i++)
        {
            MatrixFunctions::Copy(input_full[i], input);
            ecapamodel.forward(input, lengths, y_1);
            Helper::print(y_1);
        }
    }

    // Testing Ecapa Classifier CDIST
    if (testMode == ecapaClassifierCdist) {
        // Cosine = 0, CDist = 1, Euclidean = 2
        ECAPA_TDNN_classifier ecapamodel(1);
        float y_1[6][6]; // Used for Cosine/Cdist

        // Load weights
        ecapamodel.loadweights("verifyCppVsPy/fullecapa_classifier_cdist.bin");

        // Metrics on Read Input will not be tested
        float input_full[100][2][64];
        float input[2][64];
        Helper::readInputs("verifyCppVsPy/testInput_ecapaClassifier_cdist_100x2x64.bin", input_full);

        float lengths = 0.4;

        for (int i = 0; i < 100; i++)
        {
            MatrixFunctions::Copy(input_full[i], input);
            ecapamodel.forward(input, lengths, y_1);
            Helper::print(y_1);
        }
    }

    // Testing Ecapa Classifier EUCLIDEAN
    if (testMode == ecapaClassifierEuclidean) {
        // Cosine = 0, CDist = 1, Euclidean = 2
        ECAPA_TDNN_classifier ecapamodel(2);
        float y_2[6]; // Used for Euclidean

        // Load weights
        ecapamodel.loadweights("verifyCppVsPy/fullecapa_classifier_euclidean.bin");

        // Metrics on Read Input will not be tested
        float input_full[100][2][64];
        float input[2][64];
        Helper::readInputs("verifyCppVsPy/testInput_ecapaClassifier_euclidean_100x2x64.bin", input_full);

        float lengths = 0.4;

        for (int i = 0; i < 100; i++)
        {
            MatrixFunctions::Copy(input_full[i], input);
            ecapamodel.forward(input, lengths, y_2);
            Helper::print(y_2);
        }
    }
    
    // Time implmentation by Chin Yi
    // Time::TestEcapa(100);
    // Time::TestBasic(100);
    // Time::TestEcapaClassifier(100);
    return 0;
}
