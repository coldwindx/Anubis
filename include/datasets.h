#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

std::pair<std::vector<std::vector<double>>, std::vector<double>> get_moons_data()
{
    std::ifstream fin("./data/moons.txt");
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    std::string line;
    double x;
    while (!fin.eof())
    {
        std::getline(fin, line);
        if (line.empty())
            continue;
        std::stringstream ss(line);

        std::vector<double> sample;
        while (ss >> x)
            sample.emplace_back(x);
        sample.pop_back();
        X.push_back(move(sample));
        Y.emplace_back(x);
    }
    fin.close();
    return make_pair(X, Y);
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> get_linear_data()
{
    std::ifstream fin("./data/linear.txt");
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    std::string line;
    double x;
    while (!fin.eof())
    {
        std::getline(fin, line);
        if (line.empty())
            continue;
        std::stringstream ss(line);

        std::vector<double> sample;
        while (ss >> x)
            sample.emplace_back(x);
        sample.pop_back();
        X.push_back(move(sample));
        Y.emplace_back(x);
    }
    fin.close();
    return make_pair(X, Y);
}