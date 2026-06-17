// verify_roundtrip.cpp — confirm the emitted retrained_class_hvs.txt still
// classifies all 10 images correctly when reloaded.
//
// Build:  g++ -O2 -std=c++11 hdc/verify_roundtrip.cpp -o hdc/verify_roundtrip
// Run:    ./hdc/verify_roundtrip
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

// take matrix + images from variables.txt
#include "../variables.txt"

static const int INPUT_DIM = 784, HD_DIM = 100, NUM_CLASSES = 10, NUM_IMAGES = 10;

static vector<float> encode(int img) {
    vector<float> hv(HD_DIM, 0.0f);
    for (int j = 0; j < HD_DIM; j++) {
        float s = 0.0f;
        for (int i = 0; i < INPUT_DIM; i++) {
            float m = matrix[i * HD_DIM + j] ? 1.0f : -1.0f;
            s += (float)images[img * INPUT_DIM + i] * m;
        }
        hv[j] = s;
    }
    return hv;
}
static void norm(vector<float>& v){ double s=0; for(float x:v)s+=(double)x*x; double m=sqrt(s); if(m>0)for(auto&x:v)x=(float)(x/m); }

// load the scaled-int prototypes emitted by mnist_retrain (×100000)
static bool load_retrained(const char* path, vector<vector<float>>& proto) {
    ifstream f(path);
    string line, all;
    while (getline(f, line)) {
        // skip comments; keep digits/commas/minus
        if (line.find("//")!=string::npos) continue;
        all += line + " ";
    }
    // parse numbers between first { and last }
    auto a = all.find('{'); auto b = all.rfind('}');
    if (a==string::npos||b==string::npos) return false;
    string body = all.substr(a+1, b-a-1);
    vector<float> vals; string tok; long t;
    for (char ch : body) { if (ch==','||ch==' '||ch=='\t'){ if(!tok.empty()){ sscanf(tok.c_str(),"%ld",&t); vals.push_back(t/100000.0f); tok.clear();} } else tok+=ch; }
    if(!tok.empty()){ sscanf(tok.c_str(),"%ld",&t); vals.push_back(t/100000.0f); }
    if ((int)vals.size() != NUM_CLASSES*HD_DIM) {
        printf("ERROR: parsed %zu values, expected %d\n", vals.size(), NUM_CLASSES*HD_DIM);
        return false;
    }
    proto.assign(NUM_CLASSES, vector<float>(HD_DIM));
    for (int c=0;c<NUM_CLASSES;c++) for(int j=0;j<HD_DIM;j++) proto[c][j]=vals[c*HD_DIM+j];
    for (auto& p : proto) norm(p);
    return true;
}

int main() {
    vector<vector<float>> proto;
    if (!load_retrained("retrained_class_hvs.txt", proto)) { printf("FAIL: could not load\n"); return 2; }

    int correct = 0;
    printf("img pred actual ok\n");
    for (int img=0; img<NUM_IMAGES; img++) {
        vector<float> hv = encode(img); norm(hv);
        int best=0; float bs=-1e9f;
        for (int c=0;c<NUM_CLASSES;c++){ float d=0; for(int j=0;j<HD_DIM;j++)d+=hv[j]*proto[c][j]; if(d>bs){bs=d;best=c;} }
        bool ok = best==img; if(ok)correct++;
        printf("%3d %4d %6d %s\n", img, best, img, ok?"ok":"WRONG");
    }
    printf("\nRound-trip accuracy: %d/%d\n", correct, NUM_IMAGES);
    return correct==NUM_IMAGES ? 0 : 1;
}
