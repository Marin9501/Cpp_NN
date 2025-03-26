#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

const double LR = 0.001;
const int BATCH_SIZE = 10;
const int EPOCHS = 100;
const int N_INPUTS = 4;
const double ELU_ALPHA = 0.01;
const double PLReLU_ALPHA = 0.0001;

typedef std::vector<std::vector<double>> layer;
typedef std::vector<std::vector<double>> matrix;
typedef std::vector<std::vector<std::vector<double>>> cache;

double ReLU(double a);
double PLReLU(double a);
double ELU(double a);
matrix matrix_element_multiply(matrix m1, matrix m2);
matrix matrix_element_sum(matrix m1, matrix m2);
matrix matrix_element_dif(matrix m1, matrix m2);
matrix mat_mul(matrix m1, matrix m2);
matrix transpose(matrix m);
matrix transpose_mult(matrix m1, matrix m2, bool ignore);
std::vector<double> MSELoss(matrix predicition, matrix expected);
std::pair<cache, cache> forward_pass(matrix in,  std::vector<std::vector<layer>> model);
void print_matrix(matrix m, std::string message);
double weighted_sum(std::vector<double> in, std::vector<double> weights);
double vector_average(std::vector<double> in);
std::vector<double> layerOut(layer layer, std::vector<double> in);
std::vector<matrix> gradient(std::vector<double> expected, cache pre_cache, cache after_cache,  std::vector<std::vector<layer>> model);
std::vector<double> column_averge(matrix in);
int train(matrix in, matrix expected, std::vector<std::vector<layer>> &model, int n_epochs);

double weigth_clip(double x){
    if (x < -10) return -10;
    if (x > 10) return 10; 
    return x;
};
double derivative(double x){
    if(x > 0){
        return 1;
    } else {
        /*if(x < -50) return std::exp(-50);*/
        /*return ELU_ALPHA*std::exp(x);*/

        return PLReLU_ALPHA;
        //return 0;

    };
};


matrix map(matrix m, std::function<double(double)> callback){
    matrix out = m;
    for(int i = 0; i < m.size(); i++){
        for(int j = 0; j < m[i].size(); j++){
            out[i][j] = callback(m[i][j]);
        };
    };
    return out;
};
matrix gen_data(int batch_size){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(-1000,1000);

    matrix data(N_INPUTS, std::vector<double>(batch_size));
    for(int i = 0; i<N_INPUTS; i++){
        for(int j = 0; j<batch_size; j++){
            data[i][j] = distr(gen);
        };
    };
    return data;
};

matrix get_expected(matrix in){
    matrix result(2, std::vector<double>(BATCH_SIZE, 0));
    for(int j = 0; j < BATCH_SIZE; j++){
        result[0][j] += in[0][j];
        result[0][j] += in[1][j];

        result[1][j] += in[2][j];
        result[1][j] += in[3][j];
    };
    return result;
};

std::vector<std::vector<layer>> gen_model(){//std::vector<int> shape){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(-1.0, 1.0);
    // std::vector<matrix> model(shape[0], matrix(shape[1], std::vector<double>(shape[2], distr(gen))));
    // for(int i = 0; i < shape[0]; i ++){
    //     for(int j = 0; j < shape[1]; j++){
    //         for(int k = 0; k < shape[2]; k++){
    //             model[i][j][k] = distr(gen);
    //         };
    //     };
    // };
    layer l1 = {{distr(gen), distr(gen), distr(gen), distr(gen)}, {distr(gen), distr(gen), distr(gen), distr(gen)}};
    layer b1 = {{distr(gen)}, {distr(gen)}};
    layer l2 = {{distr(gen), distr(gen)}, {distr(gen), distr(gen)}};
    layer b2 = {{distr(gen)}, {distr(gen)}};
    std::vector<std::vector<layer>> model;
    model.insert(model.begin(), {b1, l1});
    model.insert(model.begin()+1, {b2, l2});

    return model;
};
int main(){
    matrix in;
    std::vector<std::vector<layer>> model;
    model = gen_model();
    for(int i = 0; i < EPOCHS; i++){
        in = gen_data(BATCH_SIZE);
        matrix expected = get_expected(in);

        // for(auto l : model){
        //     print_matrix(l[0], "Weights: ");
        //     print_matrix(l[1], "Weights: ");
        // };
        train(in, expected, model, i);
    };

    std::cout<< "---TEST---\n";

    in = gen_data(1);
    print_matrix(in, "In: ");

    std::vector<matrix> prediction = forward_pass(in, model).second;
    for( matrix p : prediction){
        print_matrix(p, "Prediction:");
    };

    matrix expected = get_expected(in);
    for(int i = 0; i < expected.size(); i++){
        std::cout << "Expected: " << expected[i][0] << "\n";
    };
    for(auto l : model){
        print_matrix(l[0], "Weights: ");
        print_matrix(l[1], "Weights: ");
    };


    return 0;
}


double vector_average(std::vector<double> in){
    return std::accumulate(in.begin(), in.end(), 0) / (double)in.size();
};

double ELU(double a){
    if (a > 0){
        return a;
    } else {
        if(a < -10) return std::exp(-10)-1;
        return ELU_ALPHA * (std::exp(a)-1);
        };
};

double ReLU(double a){
    return std::max(0.0, a);
};

double PLReLU(double a){
    if (a > 0){
        return a;
    } else {
        return PLReLU_ALPHA * a;
        };
};


std::vector<double> MSELoss(matrix prediction, matrix expected){
    matrix temp = map(matrix_element_dif(prediction, expected), [](double x) -> double {return pow(x, 2);});
    std::vector<double> ans(prediction.size());
    for(int i = 0; i < prediction.size(); i++){
        ans[i] = vector_average(temp[i]);
    };
    return ans;
};


double weighted_sum(std::vector<double> in, std::vector<double> weights){
    double ans = 0;
    for(int i = 0; i < weights.size(); i++){
        ans += in[i] * weights[i];
    };
    return ans;
};


std::vector<double> layerOut(layer layer, std::vector<double> in){

    in.insert(in.begin(), 1);
    std::vector<double> ans(layer.size(), 0);
    for(int i = 0; i < layer.size(); i++){
        ans[i] = PLReLU(weighted_sum(in, layer[i]));
    }

    return ans;
};


std::vector<std::vector<matrix>> gradient(matrix expected, cache pre_cache, cache after_cache,  std::vector<std::vector<layer>> model){
    matrix dif;
    std::vector<std::vector<layer>> grad = model;
    std::vector<layer> loss(model.size());

    for(int i = model.size()-1; i >= 0; i--){
        if(i == model.size()-1){
            dif = matrix_element_dif(after_cache[i+1], expected);
            loss[i] = matrix_element_multiply(matrix_element_multiply(dif, matrix(dif.size(), std::vector<double>(dif[0].size(), (double)1/BATCH_SIZE))), map(pre_cache[i+1], derivative));
        } else { 
            loss[i] = matrix_element_multiply(mat_mul(transpose(model[i+1][1]), loss[i+1]), map(pre_cache[i+1], derivative));
        };

        grad[i][1] = mat_mul(loss[i], transpose(after_cache[i]));
        for(int j = 0; j < grad[i][0].size(); j++){
            grad[i][0][j][0] = std::accumulate(loss[i][j].begin(), loss[i][j].end(), 0);
        };
    };
    return grad;
};


std::vector<double> column_averge(matrix in){
    std::vector<double> averages(in.size());
    for(int i = 0; i < in[0].size(); i++){
        double sum = 0;
        for(int j = 0; j < in.size(); j++){
            sum += in[j][i];
        };
        averages[i] = sum / in.size();
    };
    return averages;
};


std::pair<cache,cache> forward_pass(matrix in, std::vector<std::vector<layer>> model){
    cache pre_cache(model.size()+1);
    cache after_cache(model.size()+1);
    pre_cache[0] = in;
    after_cache[0] = in;
    matrix after_mul;
    matrix ext_bias;
    for(int i = 0; i < model.size(); i ++){
        after_mul = mat_mul(model[i][1], after_cache[i]);
        
        ext_bias.resize(model[i][0].size());
        for(int j = 0; j < model[i][0].size(); j++){
            ext_bias[j] = std::vector<double>(after_mul[0].size(), model[i][0][j][0]);
        };
        pre_cache[i+1] = matrix_element_sum(ext_bias, after_mul);
        after_cache[i+1] = map(pre_cache[i+1], PLReLU);
    };


    return std::pair(pre_cache, after_cache);
};

int train(matrix in, matrix expected, std::vector<std::vector<layer>> &model, int n_epochs){
    // print_matrix(in, "in:");
    // print_matrix(expected, "expected:");
    cache pre_cache;
    cache after_cache;

    std::pair<cache, cache> cache = forward_pass(in, model);
    pre_cache = cache.first;
    after_cache = cache.second;
    // for( matrix l : pre_cache){
    //     print_matrix(l, "pre cache:");
    // };
    // for( matrix l : after_cache){
    //     print_matrix(l, "after cache:");
    // };
   
    if (n_epochs % 200 == 0){
        print_matrix({MSELoss(after_cache[after_cache.size()-1], expected)}, "Loss: ");
    };
    
    std::vector<std::vector<matrix>> grad = gradient(expected, pre_cache, after_cache, model);
    // for(auto g : grad){
    //     print_matrix(g[0], "GRAD: ");
    //     print_matrix(g[1], "GRAD: ");
    // };

    //update weights
    for(int i = 0; i < model.size(); i ++){
        for(int j = 0; j < model[i].size(); j ++){
            for(int k = 0; k < model[i][j].size(); k ++){
                for(int l = 0; l < model[i][j][k].size(); l++){
                    model[i][j][k][l] -= grad[i][j][k][l] * LR;
                };
            };
        };
    };

    return 0;
};

void print_matrix(matrix m, std::string message){
    std::cout << "\n" << message << "\n";
    for(int i = 0; i < m.size(); i++){
        for(int j = 0; j < m[i].size(); j++){
            std::cout << m[i][j] << " ";
        };
        std::cout << "\n";
    };
    std::cout << "\n";
};

matrix matrix_element_multiply(matrix m1, matrix m2){
    matrix ans(m1.size(), std::vector<double>(m1[0].size()));
    for(int i = 0; i < m1.size(); i++){
        for(int j = 0; j < m1[i].size(); j ++){
            ans[i][j] = m1[i][j] * m2[i][j];
        };
    };
    return ans;
};

matrix matrix_element_sum(matrix m1, matrix m2){
    matrix ans = m1;
    for(int i = 0; i < m2.size(); i ++){
        for(int j = 0; j < m2[i].size(); j ++){
            ans[i][j] += m2[i][j];
        };
    };
    return ans;
};

matrix matrix_element_dif(matrix m1, matrix m2){
    matrix ans = m1;
    for(int i = 0; i < m2.size(); i ++){
        for(int j = 0; j < m2[i].size(); j ++){
            ans[i][j] -= m2[i][j];
        };
    };
    return ans;
};

matrix transpose_mult(matrix m1, matrix m2, bool ignore){
    matrix ans(m1[0].size(), std::vector<double>(m2[0].size(), 0));
    for(int i = 0; i < m1.size(); i ++){
        for(int j = 0; j < m1[0].size(); j++){
            for(int k = 0; k < m2[i].size(); k++){
                if (ignore){
                    ans[j][k] += m1[i][j+1] * m2[i][k];
                } else {
                    ans[j][k] += m1[i][j] * m2[i][k];
                };
            };
        };
    };
    if (ignore){
        ans.erase(ans.end());
    };
    return ans;
};

matrix mat_mul(matrix m1, matrix m2){
    matrix ans(m1.size(), std::vector<double>(m2[0].size()));
    for(int i = 0; i < m1.size(); i++){
        for(int j = 0; j < m2[0].size(); j++){
            for(int k = 0; k < m1[i].size(); k++){
                ans[i][j] += m1[i][k] * m2[k][j];
            };
        };
    };
    return ans;
};


matrix transpose(matrix m){
    matrix ans(m[0].size(), std::vector<double>(m.size(), 0));
    for(int i = 0; i < m.size(); i++){
        for(int j = 0; j < m[i].size(); j++){
            ans[j][i] = m[i][j];
        };
    };
    return ans;
};
