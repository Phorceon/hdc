#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

#include "../variables.txt"

static const int INPUT_DIM    = 784;
static const int HD_DIM       = 100;
static const int NUM_CLASSES  = 10;
static const int MAX_EPOCHS   = 100;
static const float TARGET_PCT = 85.0f;

static unsigned swap_endian(unsigned x) {
    return ((x >> 24) & 0xFF) |
           ((x >> 8)  & 0xFF00) |
           ((x << 8)  & 0xFF0000) |
           ((x << 24) & 0xFF000000);
}

static void load_mnist_images(const char* path, vector<unsigned char>& images, int& count) {
    FILE* f = fopen(path, "rb");
    unsigned magic, num, rows, cols;
    fread(&magic, 4, 1, f); magic = swap_endian(magic);
    fread(&num,   4, 1, f); num   = swap_endian(num);
    fread(&rows,  4, 1, f); rows  = swap_endian(rows);
    fread(&cols,  4, 1, f); cols  = swap_endian(cols);
    count = num;
    images.resize(num * INPUT_DIM);
    fread(images.data(), 1, num * INPUT_DIM, f);
    fclose(f);
}

static void load_mnist_labels(const char* path, vector<unsigned char>& labels, int& count) {
    FILE* f = fopen(path, "rb");
    unsigned magic, num;
    fread(&magic, 4, 1, f); magic = swap_endian(magic);
    fread(&num,   4, 1, f); num   = swap_endian(num);
    count = num;
    labels.resize(num);
    fread(labels.data(), 1, num, f);
    fclose(f);
}

static vector<float> encode(const vector<unsigned char>& images, int img_idx) {
    vector<float> hv(HD_DIM, 0.0f);
    for (int j = 0; j < HD_DIM; j++) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_DIM; i++) {
            float m = matrix[i * HD_DIM + j] == 0 ? -1.0f : 1.0f;
            sum += (float)images[img_idx * INPUT_DIM + i] * m;
        }
        hv[j] = sum;
    }
    return hv;
}

static void normalize_f(vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += (double)x * x;
    double mag = sqrt(s);
    if (mag > 0.0) for (float& x : v) x = (float)(x / mag);
}

static int classify(const vector<float>& hv,
                    const vector<vector<float>>& proto) {
    int best = 0; float bs = -1e9f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        float dot = 0.0f, pmag = 0.0f;
        for (int j = 0; j < HD_DIM; j++) {
            dot  += hv[j] * proto[c][j];
            pmag += proto[c][j] * proto[c][j];
        }
        if (pmag > 0.0f) dot /= sqrt(pmag);
        if (dot > bs) { bs = dot; best = c; }
    }
    return best;
}

static int retrain_one_epoch(const vector<vector<float>>& encoded,
                              const vector<int>& labels,
                              const vector<int>& order,
                              int start_offset,
                              vector<vector<float>>& proto) {
    int updates = 0;
    int n = (int)order.size();
    for (int k = 0; k < n; k++) {
        int i = order[(k + start_offset) % n];
        int pred = classify(encoded[i], proto);
        int true_class = labels[i];
        if (pred != true_class) {
            for (int j = 0; j < HD_DIM; j++) {
                proto[pred][j]       -= encoded[i][j];
                proto[true_class][j] += encoded[i][j];
            }
            updates++;
        }
    }
    for (int c = 0; c < NUM_CLASSES; c++) normalize_f(proto[c]);
    return updates;
}

static int evaluate(const vector<vector<float>>& encoded,
                    const vector<int>& labels,
                    const vector<vector<float>>& proto) {
    int correct = 0;
    for (size_t i = 0; i < encoded.size(); i++) {
        if (classify(encoded[i], proto) == labels[i]) correct++;
    }
    return correct;
}

int main() {
    printf("HDC training from scratch with target-accuracy early stopping\n");
    printf("  Using professor's projection matrix from variables.txt\n");
    printf("  max_epochs=%d  target_accuracy=%.1f%% (on validation)\n\n",
           MAX_EPOCHS, TARGET_PCT);

    vector<unsigned char> train_imgs, test_imgs;
    vector<unsigned char> train_lbls, test_lbls;
    int tr_n, tl_n, te_n, tel_n;
    load_mnist_images("mnist_data/train-images.idx3-ubyte", train_imgs, tr_n);
    load_mnist_labels("mnist_data/train-labels.idx1-ubyte", train_lbls, tl_n);
    load_mnist_images("mnist_data/t10k-images.idx3-ubyte",  test_imgs,  te_n);
    load_mnist_labels("mnist_data/t10k-labels.idx1-ubyte",  test_lbls, tel_n);
    printf("Loaded %d train images, %d test images\n\n", tr_n, te_n);

    vector<vector<float>> tr_enc, val_enc;
    vector<int> tr_y, val_y;
    for (int i = 0; i < tr_n; i++) {
        vector<float> hv = encode(train_imgs, i);
        normalize_f(hv);
        if (i % 5 == 0) { val_enc.push_back(hv); val_y.push_back(train_lbls[i]); }
        else            { tr_enc.push_back(hv); tr_y.push_back(train_lbls[i]); }
    }
    int train_size = (int)tr_enc.size();
    int val_size   = (int)val_enc.size();
    printf("Split: %d train, %d validation, %d test\n\n", train_size, val_size, te_n);

    vector<vector<float>> proto(NUM_CLASSES, vector<float>(HD_DIM, 0.0f));
    for (int i = 0; i < train_size; i++) {
        int c = tr_y[i];
        for (int j = 0; j < HD_DIM; j++) proto[c][j] += tr_enc[i][j];
    }
    for (int c = 0; c < NUM_CLASSES; c++) normalize_f(proto[c]);

    printf("epoch  train_acc            val_acc\n");
    int val_correct = evaluate(val_enc, val_y, proto);
    printf("%4d  %5d/%d = %5.1f%%   %5d/%d = %5.1f%%\n",
           0, evaluate(tr_enc, tr_y, proto), train_size,
           100.0f * evaluate(tr_enc, tr_y, proto) / train_size,
           val_correct, val_size, 100.0f * val_correct / val_size);

    vector<int> order(train_size);
    for (int i = 0; i < train_size; i++) order[i] = i;

    int epoch;
    for (epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
        int offset = (epoch * 481) % train_size;
        int updates = retrain_one_epoch(tr_enc, tr_y, order, offset, proto);

        int tr_correct  = evaluate(tr_enc,  tr_y,  proto);
        val_correct     = evaluate(val_enc, val_y, proto);

        printf("%4d  %5d/%d = %5.1f%%   %5d/%d = %5.1f%%  (updates=%d)",
               epoch, tr_correct, train_size, 100.0f * tr_correct / train_size,
               val_correct, val_size, 100.0f * val_correct / val_size, updates);

        if (100.0f * val_correct / val_size >= TARGET_PCT) {
            printf("  * TARGET REACHED\n");
            break;
        }
        printf("\n");
    }
    printf("\nEARLY STOP @ epoch %d\n\n", epoch);

    printf("Final evaluation on %d-image MNIST test set...\n", te_n);
    vector<vector<float>> te_enc(te_n);
    vector<int> te_y(te_n);
    for (int i = 0; i < te_n; i++) {
        te_enc[i] = encode(test_imgs, i);
        normalize_f(te_enc[i]);
        te_y[i] = test_lbls[i];
    }

    vector<int> class_correct(NUM_CLASSES, 0);
    vector<int> class_total(NUM_CLASSES, 0);
    int total_correct = 0;
    for (int i = 0; i < te_n; i++) {
        int pred = classify(te_enc[i], proto);
        int actual = te_y[i];
        class_total[actual]++;
        if (pred == actual) { class_correct[actual]++; total_correct++; }
    }

    printf("\nPer-class accuracy:\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        printf("  Class %d: %5d/%5d = %5.1f%%\n",
               c, class_correct[c], class_total[c],
               100.0f * class_correct[c] / class_total[c]);
    }

    float ours_pct = 100.0f * total_correct / te_n;
    printf("\n-----------------------------------------------\n");
    printf("Professor's prototypes (cited):   7070/10000 = 70.7%%\n");
    printf("Ours (early stop @ epoch %d):    %d/%d = %.1f%%\n",
           epoch, total_correct, te_n, ours_pct);
    printf("Delta: %+.1f percentage points\n", ours_pct - 70.7f);
    printf("-----------------------------------------------\n");

    return 0;
}