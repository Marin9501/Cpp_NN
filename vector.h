#ifndef MATRIX_H
#define MATRIX_H

#include <array>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>


//PRINT ND VECTOR
// Base case
template <typename T>
void PrintArray(const T& value) {
    std::cout << value << " ";
}

template <typename T>
void PrintArray(const std::vector<T> &vec) {
    for (const auto& element : vec) {
        PrintArray(element);
    };
    std::cout << "\n";
}

template <typename T>
void NDArray(const std::vector<T> &vec, std::string message = ""){
    std::cout << message << "\n";
    PrintArray(vec);
};


void MatrixSymmetry(std::vector<std::vector<double>> &m){
    //Safety check for symmetry
    for(int i = 1; i < m.size(); i++){
        if (m[i].size() != m[i-1].size()){
            throw std::runtime_error("Matrix is not symetrical when doing matmul");
        };
    };
    return;
};

//MATMUL
std::vector<std::vector<double>> MatMul(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2){
    std::vector<std::vector<double>> mOut(m1.size(), std::vector<double>(m2[0].size(), 0));
    //Check their symmetry
    MatrixSymmetry(m1);
    MatrixSymmetry(m2);
    //Safety check for dimensions
    if(m1[0].size() != m2.size()){
        throw std::runtime_error("Missmatched matrix dimensions when doing matmul");
    };

    for(int i = 0; i < m1.size(); i++){
        for(int j = 0; j < m2[0].size(); j++){
            for(int k = 0; k < m2.size(); k++){
                mOut[i][j] += m1[i][k] * m2[k][j];
            };
        };
    };

    return mOut;
};

//Matrix element addintion
std::vector<std::vector<double>> ElementAddition(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2){
    MatrixSymmetry(m1);
    MatrixSymmetry(m2);
    std::vector<std::vector<double>> out = m1;
    for(int i = 0; i < m2.size(); i++){
        for(int j = 0; j < m2[i].size(); j++){
            out[i][j] += m2[i][j];
        };
    };
    return out;
};

//Matrix element subtraction
std::vector<std::vector<double>> ElementSubraction(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> &m2){
    MatrixSymmetry(m1);
    MatrixSymmetry(m2);
    for(int i = 0; i < m2.size(); i++){
        for(int j = 0; j < m2[i].size(); j++){
            m1[i][j] -= m2[i][j];
        };
    };
    return m1;
};

//Matrix element multiplication
std::vector<std::vector<double>> ElementMultiply(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> &m2){
    MatrixSymmetry(m1);
    MatrixSymmetry(m2);
    for(int i = 0; i < m2.size(); i++){
        for(int j = 0; j < m2[i].size(); j++){
            m1[i][j] *= m2[i][j];
        };
    };
    return m1;
};

//ScalarMult
std::vector<std::vector<double>> ScalarMult(std::vector<std::vector<double>> m, int k){
    for(int i = 0; i < m.size(); i++){
        for(int j = 0; j < m[i].size(); j++){
            m[i][j] *= k;
        };
    };
    return m;
};

//Matrix transpose
std::vector<std::vector<double>> Transpose(std::vector<std::vector<double>> m){
    MatrixSymmetry(m);
    std::vector<std::vector<double>> out(m[0].size(), std::vector<double>(m.size()));
    for(int i = 0; i < m.size(); i++){
        for(int j = 0; j < m[0].size(); j++){
            out[j][i] = m[i][j];
        };
    };
    return out;
};

//Matrix average dim 1
std::vector<double> MatrixAverageD1(std::vector<std::vector<double>> &m){
    std::vector<double> out(m[0].size(), 0);
    for(int j = 0; j < m[0].size(); j++){
        for(int i = 0; i < m.size(); i++){
            out[j] += m[i][j];
        };
        out[j] /= m.size();
    };
    return out;
};

#endif
