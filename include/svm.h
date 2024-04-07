#pragma once

#include <vector>
#include <functional>

namespace clearn
{
    class SVC
    {
    public:
        using kernel = std::function<double(std::vector<double> &, std::vector<double> &)>;

    public:
        SVC(int n_features_in, int max_epochs, double C, double toler, kernel k);
        void fit(std::vector<std::vector<double>> &X, std::vector<double> &Y);
        std::vector<double> getW() const;
        double getB() const;

    private:
        int n_features_in;
        int n_samples;
        int max_epochs;
        double C;
        double toler;
        double b;
        std::vector<double> W;
        std::vector<double> alpha;
        std::vector<std::pair<bool, double>> E;
        kernel K;
        std::vector<std::vector<double>> *X;
        std::vector<double> *Y;

    private:
        /**
         * 计算每个样本点k的Ek值，就是计算误差值=预测值-标签值
         */
        double _calc_Ek_(int k);
        /**
         * 内循环的启发式方法，获取最大差值|Ei-Ej|对应的Ej的索引j
         */
        std::pair<int, double> _select_j_(const int i, double Ei);

        /**
         * 内循环
         */
        int _inner_(int i);
        /**
         * 随机选择一个j作为alpha_2的下标
         */
        int _select_rand_j_(int i);
    };
}