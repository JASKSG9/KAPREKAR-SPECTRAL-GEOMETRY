// d6_enum_final.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <chrono>
using namespace std;

constexpr int D = 6, BASE = 10, MAXN = 1000000;

int kaprekar_step(int n) {
    array<int, D> d{};
    for (int i = 0; i < D; ++i) { d[i] = n % BASE; n /= BASE; }
    sort(d.begin(), d.end());
    int asc = 0, desc = 0;
    for (int i = 0; i < D; ++i) { asc = asc * BASE + d[i]; desc = desc * BASE + d[D-1-i]; }
    return desc - asc;
}

bool is_repdigit(int n) {
    int r = n % BASE;
    for (int i = 1; i < D; ++i) { n /= BASE; if (n % BASE != r) return false; }
    return true;
}

int main() {
    vector<int> depth(MAXN, 0);
    int maxd = 0, total = 0;
    #pragma omp parallel for reduction(max:maxd) reduction(+:total)
    for (int n = 0; n < MAXN; ++n) {
        if (is_repdigit(n)) continue;
        total++;
        int x = n, steps = 0;
        bool seen[100] = {false};
        while (steps < 100 && !seen[x % 100]) {
            seen[x % 100] = true;
            x = kaprekar_step(x);
            steps++;
            if (x == 0 || x == 495 || x == 6174) break;
        }
        depth[n] = steps;
        if (steps > maxd) maxd = steps;
    }
    cout << "τ_max = " << maxd << "
";
    vector<int> hist(maxd+1,0);
    for (int n=0; n<MAXN; ++n) if(depth[n]>0) hist[depth[n]]++;
    for (int t=1; t<=maxd; ++t) if(hist[t]) cout << "τ=" << t << ": " << hist[t] << "
";
    return 0;
}
