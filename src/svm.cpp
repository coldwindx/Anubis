#include <cstdio>
#include "svm.h"

// 计算符号位
#define SignBit(x) (((signed char *)&x)[sizeof(x) - 1] >> 7 | 1)
#define EPSINON 1e-6

using std::function;
using std::min, std::max, std::abs;
using std::vector;

clearn::SVC::SVC(int n_features_in, int max_epochs, double C, double toler, kernel k) : n_features_in(n_features_in), C(C), toler(toler), b(0.0), max_epochs(max_epochs), K(k)
{
    W.resize(n_features_in, 0.0);
}

void clearn::SVC::fit(std::vector<std::vector<double>> &X, std::vector<double> &Y)
{
    this->X = &X;
    this->Y = &Y;
    this->n_samples = X.size();
    // 1. initialize paramster
    alpha.resize(n_samples, 0.0);
    E.resize(n_samples, std::make_pair<bool, double>(false, 0.0));
    // for (int i = 0; i < n_samples; ++i)
    //     E[i] = _calc_Ek_(i);

    // 2. SMO algorithm
    bool entire = true; // 标志是否应该遍历整个数据集
    int change = 0;
    for (int epoch = 0; epoch < max_epochs && (0 < change || entire); ++epoch)
    {
        // printf("epoch: %d\n", epoch);
        change = 0;
        if (entire)
        {
            for (int i = 0; i < n_samples; ++i)
                change += _inner_(i);
            printf("full dataset, iter: %d ,pairs changed:%d\n", epoch, change);
        }
        else
        {
            for (int i = 0; i < n_samples; ++i)
            {
                if (0.0 < alpha[i] && alpha[i] < C)
                {
                    change += _inner_(i);
                }
            }
            printf("non bound, iter: %d ,pairs changed:%d\n", epoch, change);
        }
        entire = entire ? false : (0 == change);
    }
    // printf("SMO finished!\n");
    // 3. 求解权重W
    for (int i = 0; i < n_samples; ++i)
        for (int j = 0; j < n_features_in; ++j)
            W[j] += alpha[i] * Y[i] * X[i][j];
}

std::vector<double> clearn::SVC::getW() const
{
    return this->W;
}

double clearn::SVC::getB() const
{
    return this->b;
}

int clearn::SVC::_inner_(int i)
{
    printf("__inner__(i = %d)\n", i);
    double Ei = _calc_Ek_(i);
    // printf("get Ei : %f\n", Ei);
    // 检查KKT条件
    if (!((*Y)[i] * Ei < -toler && alpha[i] < C || (*Y)[i] * Ei > toler && alpha[i] > 0))
    {
        printf("Un KKT\n");
        return 0;
    }
    // 选取差值最大的alpha_2索引
    auto [j, Ej] = _select_j_(i, Ei);
    printf("\tselect j = %d, Ej = %f\n", j, Ej);
    double old_alpha_i = alpha[i], old_alpha_j = alpha[j];
    // printf("a1 = %f, a2 = %f\n", old_alpha_i, old_alpha_j);
    // 约束条件
    // double L = SignBit((*Y)[i]) == SignBit((*Y)[j]) ? max(0.0, old_alpha_i + old_alpha_j - C) : max(0.0, old_alpha_j - old_alpha_i);
    // double H = SignBit((*Y)[i]) == SignBit((*Y)[j]) ? min(C, old_alpha_i + old_alpha_j) : min(C, C + old_alpha_j - old_alpha_i);

    double L, H;
    if ((*Y)[i] * (*Y)[j] < 0.0)
    {
        L = max(0.0, old_alpha_j - old_alpha_i);
        H = min(C, C + old_alpha_j - old_alpha_i);
    }
    else
    {
        L = max(0.0, old_alpha_i + old_alpha_j - C);
        H = min(C, old_alpha_i + old_alpha_j);
    }
    printf("L = %f, H = %f\n", L, H);
    if (abs(L - H) < EPSINON)
    {
        printf("L == H, L: %f, H: %f\n", L, H);
        return 0;
    }

    // 计算eta = K11 + K22 - 2K12
    double eta = K((*X)[i], (*X)[i]) + K((*X)[j], (*X)[j]) - 2.0 * K((*X)[i], (*X)[j]);
    printf("eta = %f\n", eta);
    if (eta <= 0.0)
    {
        printf("eta <= 0\n");
        return 0;
    }

    // 修改alpha2，更新Ek
    alpha[j] = min(max(alpha[j] + (*Y)[j] * (Ei - Ej) / eta, L), H);
    E[j].first = true, E[j].second = _calc_Ek_(j);
    printf("\taj = %f, Ej = %f\n", alpha[j], E[j].second);
    // 检查alpha2的变化量
    if (abs(alpha[j] - old_alpha_j) < EPSINON)
    {
        printf("J not move enough\n");
        return 0;
    }
    // 跟新alpha1和Ek
    alpha[i] += (*Y)[i] * (*Y)[j] * (old_alpha_j - alpha[j]);
    E[i].first = true, E[i].second = _calc_Ek_(i);
    printf("\tai = %f, Ei = %f\n", alpha[i], E[i].second);
    // 更新阈值b
    double b1 = b - Ei - (*Y)[i] * (alpha[i] - old_alpha_i) * K((*X)[i], (*X)[i]) - (*Y)[j] * (alpha[j] - old_alpha_j) * K((*X)[i], (*X)[j]);
    double b2 = b - Ej - (*Y)[i] * (alpha[i] - old_alpha_i) * K((*X)[i], (*X)[j]) - (*Y)[j] * (alpha[j] - old_alpha_j) * K((*X)[j], (*X)[j]);
    b = 0.0 < alpha[i] && alpha[i] < C ? b1 : (0 < alpha[j] && alpha[j] < C ? b2 : (b1 + b2) / 2.0);
    return 1;
}

std::pair<int, double> clearn::SVC::_select_j_(const int i, double Ei)
{
    // printf("__select_j__\n");
    int ansj = n_samples - 1;
    double delta = 0.0, Ej = 0.0;
    E[i].first = true, E[i].second = Ei;
    // 获取所有 具有有效E的索引
    vector<int> valids;
    for (int k = 0; k < n_samples; ++k)
    {
        if (E[k].first)
            valids.emplace_back(k);
    }
    // 没有有效E时(必定包含Ei)，随机返回一个索引
    if (valids.size() <= 1)
    {
        ansj = _select_rand_j_(i);
        Ej = _calc_Ek_(ansj);
        // printf("select rand j %d, Ej = %f\n", ansj, Ej);
        return std::make_pair(ansj, Ej);
    }
    // 有有效E时，选择|Ei-Ej|最大的索引
    for (auto &k : valids)
    {
        if (i == k)
            continue;
        double Ek = _calc_Ek_(k);
        // // printf("__calc_Ek__ k %d, Ek = %f\n", k, Ek);
        double d = abs(Ei - Ek);
        if (i == 3)
        {
            printf("\tk = %d, d = %f\n", k, d);
        }
        if (delta < d)
        {
            ansj = k;
            delta = d;
            Ej = Ek;
        }
    }
    return std::make_pair(ansj, Ej);
}

int clearn::SVC::_select_rand_j_(int i)
{
    int j = i;
    while (i == j)
        j = rand() % n_samples;
    j = 73;
    printf("__select_rand_j__ j = %d\n", j);
    return j;
}

double clearn::SVC::_calc_Ek_(int k)
{
    double ans = b - (*Y)[k];
    for (int j = 0; j < n_samples; ++j)
        ans += alpha[j] * (*Y)[j] * K((*X)[k], (*X)[j]);
    return ans;
}
