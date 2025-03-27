#include "vector.h"
#include "model.h"
#include <cmath>
#include <numeric>
#include <vector>

const int BATCH_SIZE = 20;
const int N_INPUTS = 3;
const int EPOCHS = 50000;
const double LR = 0.005;
const double EPSILON = 0.00000001;

std::vector<std::vector<double>> GetExpected(std::vector<std::vector<double>> data);

int main(){
    //model({input_size, neurons in layer...}, (min inicialized val, max inicialized val), Batch size
    Model<4> model(std::vector<int>{N_INPUTS,30,1}, std::vector<Functions>{PReLU,Ident}, std::pair<double, double>(0, 1));

    for(int t = 1; t < EPOCHS+1; t++){
        //Shape: Batchsize x nInputs
        std::vector<std::vector<double>> data = GenData(std::pair<int, int>(BATCH_SIZE, N_INPUTS), std::pair<double, double>(0, 10));
        std::vector<std::vector<double>> expected = GetExpected(data);

        std::vector<std::vector<double>> predicted = model.ForwardPass(data);
        if(t % 100 == 0){
            std::cout << "Loss: " << MSELoss(predicted, expected) << "\n";
        };
        model.Gradient(expected, predicted);
        model.UpdateGradient(LR, t, EPSILON);
    };
    //Shape: Batchsize x nInputs
    std::vector<std::vector<double>> data = GenData(std::pair<int, int>(1, N_INPUTS), std::pair<double, double>(0, 10));
    NDArray(data, "Data:");
    std::vector<std::vector<double>> expected = GetExpected(data);
    NDArray(expected, "EXPECTED");
    NDArray(model.ForwardPass(data), "PREDICTED: ");

    NDArray(model.GetWeights(), "weights");
    NDArray(model.GetBiases(), "Biases: ");

    return 0;
};

//Cusotm output function 
std::vector<std::vector<double>> GetExpected(std::vector<std::vector<double>> data){
    std::vector<std::vector<double>> out;
    for(int i = 0; i < data.size(); i++){
        double temp = std::pow(data[i][0], 3)+ pow(2, data[i][1]/10) - data[i][2];
        out.push_back({temp});
    };
    return out;
};
