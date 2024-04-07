#include "matplotlibcpp.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include "datasets.h"
#include "svm.h"
namespace plt = matplotlibcpp;
using namespace std;
using namespace clearn;
// g++ -std=c++11 -W -o main main.cpp -I /home/zhulin/anaconda3/envs/torch/include/python3.8 -I /home/zhulin/anaconda3/envs/torch/lib/python3.8/site-packages/numpy/core/include -I./include -L /home/zhulin/anaconda3/envs/torch/lib -lpython3.8 -lpthread
int main()
{
    auto [X, Y] = get_linear_data();

    SVC::kernel k = [&](std::vector<double> &x, std::vector<double> &y)
    {
        double ans = 0.0;
        for (int i = 0, n = x.size(); i < n; ++i)
            ans += x[i] * y[i];
        return ans;
    };

    SVC *svc = new SVC(2, 100, 0.6, 0.001, k);
    svc->fit(X, Y);

    Py_SetPythonHome(L"/home/zhulin/anaconda3/envs/torch");
    vector<double> cls_1x, cls_1y, cls_2x, cls_2y;
    for (int i = 0; i < X.size(); ++i)
    {
        if (abs(Y[i] - 1.0) < 1e-6)
        {
            cls_1x.push_back(X[i][0]);
            cls_1y.push_back(X[i][1]);
        }
        else
        {
            cls_2x.push_back(X[i][0]);
            cls_2y.push_back(X[i][1]);
        }
    }
    std::map<std::string, std::string> k1 = {{"c", "b"}};
    plt::scatter(cls_1x, cls_1y, 10.0, {{"c", "b"}});
    plt::scatter(cls_2x, cls_2y, 10.0, {{"c", "r"}});

    vector<double> W = svc->getW();
    double b = svc->getB();
    vector<double> xx, yy1, yy2, yy3;
    for (double i = 0; i < 10.0; i += 0.1)
    {
        xx.push_back(i);
        yy1.push_back((-W[0] * i - b) / W[1]);
        yy2.push_back((-W[0] * i - b - 1) / W[1]);
        yy3.push_back((-W[0] * i - b + 1) / W[1]);
    }
    plt::plot(xx, yy1);
    plt::plot(xx, yy2);
    plt::plot(xx, yy3);
    plt::save("f.png");
    plt::show();
    return 0;
}
int test()
{
    // export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhulin/anaconda3/envs/torch/lib
    Py_SetPythonHome(L"/home/zhulin/anaconda3/envs/torch");
    plt::plot({1, 3, 2, 4});
    plt::pause(2); // 最好加上该句，否则有时候显示不了图像，或者图像显示很慢

    plt::save("f.png");
    plt::show();
    return 0;
}
