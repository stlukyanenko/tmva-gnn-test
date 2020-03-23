#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <json.hpp>

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

// Converts an std::vector of std::vector to Eigen Matrix
template <typename T>
Matrix<T, Dynamic, Dynamic> vv_to_matrix(vector<vector<T>> vv){
    Matrix <T, Dynamic, Dynamic>  m (vv.size(), vv[0].size());
    for (int i = 0; i< vv.size(); i++){
        for (int j = 0; j < vv[i].size(); j++){
            m(i,j)= vv[i][j];
        }
    }
    return m;
}


// Converts a std::vector to Eigen Vector
template <typename T>
Vector<T,Dynamic> std_vector_to_eigen(vector <T> v){
    Vector<T,Dynamic> ev (v.size());
    for (int i = 0; i< v.size(); i++){
        ev(i)= v[i];
    }
    return ev;
}

VectorXd ReLu(VectorXd x){
    for (int i = 0; i < x.size(); i++)
        if (x(i) < 0) x(i) = 0;
    return x;
}

class DoubleLayerMLP{
    private:
        MatrixXd w1;
        VectorXd b1;
        MatrixXd w2;
        VectorXd b2;
    public:
        
        DoubleLayerMLP(MatrixXd w1, VectorXd b1, MatrixXd w2, VectorXd b2){
            this->w1 = w1;      
            this->b1 = b1;
            this->w2 = w2;
            this->b2 = b2;
        }

        DoubleLayerMLP(const DoubleLayerMLP & nn){
            this->w1 = nn.w1;      
            this->b1 = nn.b1;
            this->w2 = nn.w2;
            this->b2 = nn.b2;
        }

        DoubleLayerMLP(){
            this->w1 = MatrixXd();      
            this->b1 = VectorXd();
            this->w2 = MatrixXd();
            this->b2 = VectorXd();
        }

        DoubleLayerMLP(json json_data, string s){
            vector<vector<double>> vv1 = json_data[s]["w1"].get<vector<vector<double>>>();
            MatrixXd w1 = vv_to_matrix(vv1);

            vector<vector<double>> vv2 = json_data[s]["w2"].get<vector<vector<double>>>();
            MatrixXd w2 = vv_to_matrix(vv2);

            vector<double> v1 = json_data[s]["b1"].get<vector<double>>();
            VectorXd b1 = std_vector_to_eigen(v1);

            vector<double> v2 = json_data[s]["b2"].get<vector<double>>();
            VectorXd b2 = std_vector_to_eigen(v2);

            this->w1 = w1;      
            this->b1 = b1;
            this->w2 = w2;
            this->b2 = b2;
        }

        VectorXd forward(VectorXd x){
            //cout<< endl << x << endl;
            //cout<< endl << w1 << endl;
            //cout<< endl << w1 * x << endl;

            return w2*ReLu(w1 * x + b1) + b2;
        }
};

class MPNN{
    private:
        int V_attributes;
        int E_attributes;
        DoubleLayerMLP edge_update_nn;
        DoubleLayerMLP vertice_update_nn;
        DoubleLayerMLP output_update_nn;
        int V_hidden;
        int E_hidden;
    public:
        MPNN(int V_attributes, int E_attributes, DoubleLayerMLP edge_update_nn, DoubleLayerMLP vertice_update_nn, DoubleLayerMLP output_update_nn, int V_hidden = 0, int E_hidden = 0){
                this -> V_attributes = V_attributes;
                this -> E_attributes = E_attributes;
                this -> edge_update_nn = edge_update_nn;
                this -> vertice_update_nn = vertice_update_nn;
                this -> output_update_nn = output_update_nn;
                this -> V_hidden = V_hidden ? V_hidden : V_attributes;
                this -> E_hidden = E_hidden ? E_hidden : E_attributes;
            }
        tuple<MatrixXd, MatrixXd, VectorXd>  forward (MatrixXd E, MatrixXi E_V, MatrixXd V){
            int E_n = E.rows();
            int V_n = V.rows();
             
            MatrixXd E_new = MatrixXd::Zero (E_n, this->E_hidden);
            MatrixXd V_new = MatrixXd::Zero (V_n, this->V_hidden);

            for (int i = 0; i < E_n; i++){
                VectorXd E_concat(V.cols() + V.cols() + E.cols());
                E_concat << V.row(E_V(i,0)).transpose(), V.row(E_V(i,1)).transpose(), E.row(i).transpose();

                //cout<<V.row(E_V(i,0)) <<endl;
                E_new.row(i) = this->edge_update_nn.forward(E_concat);
            }

            VectorXd V_agregated = VectorXd::Zero(this->V_hidden);

            for (int i = 0; i < V_n; i++){  
                VectorXd E_agregated = VectorXd::Zero(this->E_hidden);
                for (int j = 0; j < E_n; j++){
                    if (E(j,1) == i)
                        E_agregated += E_new.row(i);
                }
                VectorXd V_concat(E_agregated.size() + V.cols());
                V_concat << E_agregated, V.row(i).transpose();
                
                V_new.row(i) = this->vertice_update_nn.forward(V_concat);
                V_agregated += V_new.row(i);
            }

            tuple<MatrixXd, MatrixXd, VectorXd> res = tuple<MatrixXd, MatrixXd, VectorXd> (E_new, V_new, this->output_update_nn.forward(V_agregated));
            return res;
        }
};


int main(int argc, char* argv[]){

    std::ifstream ifs ("weights.json", std::ifstream::in);
    json json_data = json::parse(ifs);
    

    DoubleLayerMLP edge_update_nn = DoubleLayerMLP(json_data, "edge_update");
    DoubleLayerMLP vertice_update_nn = DoubleLayerMLP(json_data, "vertice_update");
    DoubleLayerMLP output_update_nn = DoubleLayerMLP(json_data, "output_update");
    MPNN model = MPNN(3, 3, edge_update_nn, vertice_update_nn, output_update_nn);
    std::ifstream graphifstream ("test_graph.json", std::ifstream::in);
    json json_graph = json::parse(graphifstream);

    vector<vector<double>> E_vv = json_graph["E"].get<vector<vector<double>>>();
    vector<vector<int>> V_E_vv = json_graph["V_E"].get<vector<vector<int>>>();
    vector<vector<double>> V_vv = json_graph["V"].get<vector<vector<double>>>();


    MatrixXd E = vv_to_matrix<double>(E_vv);
    MatrixXi V_E = vv_to_matrix<int> (V_E_vv);
    MatrixXd V = vv_to_matrix<double>(V_vv);

    tuple<MatrixXd, MatrixXd, VectorXd> res = model.forward(E,V_E,V);
    MatrixXd new_E = get<0> (res);
    MatrixXd new_V = get<1> (res);
    MatrixXd new_u = get<2> (res);
    cout << "Edge attributes: "<< endl << new_E  << endl;
    cout << "Vertice attributes: "<< endl << new_V << endl;
    cout << "Global attributes: "<< endl << new_u << endl;
}