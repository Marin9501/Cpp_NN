#ifndef MODEL_H
#define MODEL_H

#include "vector.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <array>
#include <vector>

//Generate random data based on shape
//Return Shape: Batchsize x nInputs
std::vector<std::vector<double>> GenData(const std::pair<int, int> shape, const std::pair<double, double> valueRange){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(valueRange.first, valueRange.second);

    std::vector<std::vector<double>> data(shape.first, std::vector<double>(shape.second, 0));
    for(int i = 0; i < shape.first; i++){
        for(int j = 0; j < shape.second; j++){
            data[i][j] = dis(gen);
        };
    };
    
    //Shape: Batchsize x nInputs
    return data;
};


//Activation functions
enum Functions {
    ReLU, 
    ELU,
    PReLU,
    Ident,
};

//MODEL
template <std::size_t N>
class Model{
    private:
        const double ELU_ALPHA = 0.1;
        const double PReLU_ALPHA = 0.01;
        int nLayers;
        //Shape: nLayers x nNeurons
        std::vector<std::vector<double>> biases;
        //Shape: nLayers x nInputs x nNeurons
        std::vector<std::vector<std::vector<double>>> weights;

        //Activation functions for each layer
        std::vector<Functions> activationFunctions;

        //Cached values from forward pass
        std::vector<std::vector<std::vector<double>>> weightedCache;
        std::vector<std::vector<std::vector<double>>> activationCache;

        //Momentum for Adam optimizer
        std::vector<std::vector<double>> mBiases;
        std::vector<std::vector<double>> vBiases;
        std::vector<std::vector<std::vector<double>>> mWeights;
        std::vector<std::vector<std::vector<double>>> vWeights;

        //Gradient vectors
        std::vector<std::vector<std::vector<double>>> weightGrad;
        std::vector<std::vector<double>> biasGrad;

        void Activation(Functions f, double &val){
            switch (f) {
                case ReLU:
                    val = val <= 0 ? 0 : val;
                    break;
                case ELU:
                    val = val <=0 ? ELU_ALPHA * (exp(val) - 1): val;
                    break;
                case PReLU:
                    val = val <= 0 ? PReLU_ALPHA * val : val;
                    break;
                case Ident:
                    val = val;
                    break;
                default:
                    val = val;
            };
            return;
        };

        //Derivative of activation functions
        std::vector<std::vector<double>> Derivative(Functions f, std::vector<std::vector<double>> &m){
            std::vector<std::vector<double>> out = m;
            switch (f) {
                case ReLU:
                    for(int i = 0; i < m.size(); i++){
                        for(int j =0; j < m[i].size(); j++){
                            out[i][j] = m[i][j] <= 0 ? 0 : 1;
                        };
                    };
                    break;
                case ELU:
                    for(int i = 0; i < m.size(); i++){
                        for(int j =0; j < m[i].size(); j++){
                            out[i][j] = m[i][j] <= 0 ? ELU_ALPHA*exp(m[i][j]) : 1;
                        };
                    };
                    break;
                case PReLU:
                    for(int i = 0; i < m.size(); i++){
                        for(int j =0; j < m[i].size(); j++){
                            out[i][j] = m[i][j] <= 0 ? PReLU_ALPHA : 1;
                        };
                    };
                    break;
                case Ident:
                    for(int i = 0; i < m.size(); i++){
                        for(int j =0; j < m[i].size(); j++){
                            out[i][j] = 1;
                        };
                    };
                    break;
                default:
                    out = m;
            };
            return out;
        };

    public:
        Model(std::vector<int> nNeurons, std::vector<Functions> F, std::pair<double, double> valueRange){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(valueRange.first, valueRange.second);

            activationFunctions = F;
            //First in nNeurons is the number of inputs 
            nLayers = nNeurons.size()-1;

            //INICIALIZE BIASES
            //Leave out the first number as it is the number of inputs
            for(int i = 1; i < nNeurons.size(); i++){
                biases.push_back(std::vector<double>(nNeurons[i], 0));
                for(int j = 0; j < nNeurons[i]; j++) biases.back()[j] = dis(gen);
            };

            //INICIALIZE WEIGHTS
            for(int i = 1; i < nNeurons.size(); i++){
                weights.push_back(std::vector<std::vector<double>>(nNeurons[i-1], std::vector<double>(nNeurons[i], 0)));
                for(int j = 0; j < nNeurons[i-1]; j++){
                    for(int k = 0; k < nNeurons[i]; k++){
                        weights.back()[j][k] = dis(gen);
                    };
                };
            };

            //Copy the weights and biases to gradient vectors so that they have the same shape
            weightGrad = weights;
            biasGrad = biases;

            //Initialize momentum vectors for Adam with zeros
            for(int i = 0; i < weights.size(); i++){
                std::vector<std::vector<double>> temp;
                for(int j = 0; j < weights[i].size(); j++){
                    temp.push_back(std::vector<double>(weights[i][j].size(), 0));
                };
                mWeights.push_back(temp);
                vWeights.push_back(temp);
            };

            for(int i = 0; i < biases.size(); i++){
                mBiases.push_back(std::vector<double>(biases[i].size(), 0));
                vBiases.push_back(std::vector<double>(biases[i].size(), 0));
            };
        };

        //FORWARD PASS
        //Biases Shape: nLayers x nNeurons
        //Weights Shape: nLayers x nInputs x nNeurons
        std::vector<std::vector<double>> ForwardPass(std::vector<std::vector<double>> input){
            //Reset cache before every epoch
            weightedCache = {};
            activationCache = {input};

            //Pass through layers
            for(int i = 0; i < weights.size(); i++){
                input = MatMul(input, weights[i]);
                std::vector<std::vector<double>> cache(input.size(), std::vector<double>(input[0].size(), 0));
                for(int j = 0; j < biases[i].size(); j++){
                    for(int k = 0; k < input.size(); k++){
                        input[k][j] += biases[i][j];
                        cache[k][j] = input[k][j];
                        Activation(activationFunctions[i], input[k][j]);
                    };
                };
                //cache the values
                weightedCache.push_back(cache);
                activationCache.push_back(input);
            };
            activationCache.erase(activationCache.end());
            return input; 
        };

        //GRADIENT CALCULATION
        void Gradient(std::vector<std::vector<double>> expected, std::vector<std::vector<double>> predicted){
            std::vector<std::vector<double>> delta;
            std::vector<std::vector<double>> derivative;
            std::vector<std::vector<double>> transpose;
            delta = ElementSubraction(predicted, expected);
            for(int i = nLayers-1; i >= 0; i--){
                derivative = Derivative(activationFunctions[i], weightedCache[i]);
                delta = ElementMultiply(delta, derivative);
                transpose = Transpose(activationCache[i]);
                weightGrad[i] = MatMul(transpose, delta);
                biasGrad[i] = MatrixAverageD1(delta);
                transpose = Transpose(weights[i]);
                delta = MatMul(delta, transpose);
            };
            return;
        };

        //Gradient update with Adam optimizer
        void UpdateGradient(const double LR, const int t, const double adamBeta1, const double adamBeta2, const double EPSILON){
            double mHat;
            double vHat;
            //Update weights
            for(int i = 0; i < nLayers; i++){
                for(int j = 0; j < weights[i].size(); j++){
                    for(int k = 0; k < weights[i][j].size(); k++){
                        mWeights[i][j][k] = adamBeta1 * mWeights[i][j][k] + (1-adamBeta1) * weightGrad[i][j][k];
                        vWeights[i][j][k] = adamBeta2 * vWeights[i][j][k] + (1-adamBeta2) * weightGrad[i][j][k] * weightGrad[i][j][k];
                        mHat = mWeights[i][j][k] / (1 - std::pow(adamBeta1, t));
                        vHat = vWeights[i][j][k] / (1 - std::pow(adamBeta2, t));

                        weights[i][j][k] -= LR * mHat / (std::sqrt(vHat) + EPSILON);
                    };
                };
            };

            //Update Biases
            for(int i = 0; i < nLayers; i++){
                for(int j = 0; j < biases[i].size(); j++){
                    mBiases[i][j] = adamBeta1 * mBiases[i][j] + (1-adamBeta1) * biasGrad[i][j];
                    vBiases[i][j] = adamBeta2 * vBiases[i][j] + (1-adamBeta2) * biasGrad[i][j] * biasGrad[i][j];
                    mHat = mBiases[i][j] / (1 - std::pow(adamBeta1, t));
                    vHat = vBiases[i][j] / (1 - std::pow(adamBeta2, t));
                    biases[i][j] -= LR * mHat / (std::sqrt(vHat) + EPSILON);
                };
            };
            return;
        };

        //Getters
        std::vector<std::vector<double>> GetBiases(){
            return biases;
        };
        std::vector<std::vector<std::vector<double>>> GetWeights(){
            return weights;
        };
        std::vector<std::vector<std::vector<double>>> GetWeightCache(){
            return weightedCache;
        };
        std::vector<std::vector<std::vector<double>>> GetActiveCache(){
            return activationCache;
        };

        std::vector<std::vector<std::vector<double>>> GetWeightGrad(){
            return weightGrad;
        }

        std::vector<std::vector<double>> GetBiasGrad(){
            return biasGrad;
        };

};

//Not really good without safety check
double MSELoss(std::vector<std::vector<double>> predicted, std::vector<std::vector<double>> expected){
    double out = 0;
    for(int i = 0; i < predicted.size(); i++){
        for(int j = 0; j < predicted[i].size(); j++){
            out += pow(expected[i][j] - predicted[i][j], 2);
        };
    };
    return out/(predicted.size()+predicted[0].size());
};

#endif
