#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <stdint.h>
using namespace std;

#include "../variables.txt"

static const int INPUT_DIM   = 784;
static const int HD_DIM      = 100;
static const int NUM_CLASSES = 10;

static uint32_t swap_endian(uint32_t x) {
    return ((x >> 24) & 0xFF) |
           ((x >> 8)  & 0xFF00) |
           ((x << 8)  & 0xFF0000) |
           ((x << 24) & 0xFF000000);
}

static void load_mnist_images(const char* path, vector<uint8_t>& images, int& count) {
    FILE* f = fopen(path, "rb");
    uint32_t magic, num, rows, cols;
    fread(&magic, 4, 1, f); magic = swap_endian(magic);
    fread(&num,   4, 1, f); num   = swap_endian(num);
    fread(&rows,  4, 1, f); rows  = swap_endian(rows);
    fread(&cols,  4, 1, f); cols  = swap_endian(cols);
    count = num;
    images.resize(num * INPUT_DIM);
    fread(images.data(), 1, num * INPUT_DIM, f);
    fclose(f);
}

static void load_mnist_labels(const char* path, vector<uint8_t>& labels, int& count) {
    FILE* f = fopen(path, "rb");
    uint32_t magic, num;
    fread(&magic, 4, 1, f); magic = swap_endian(magic);
    fread(&num,   4, 1, f); num   = swap_endian(num);
    count = num;
    labels.resize(num);
    fread(labels.data(), 1, num, f);
    fclose(f);
}

static vector<float> encode(const vector<uint8_t>& images, int img_idx) {
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

static string check_early_stop(int val_correct, int val_total, float target_pct) {
    float acc = (float)val_correct / val_total * 100.0f;
    if (acc >= target_pct) {
        return "target accuracy " + to_string((int)target_pct) +
               "% reached on validation";
    }
    return "";
}

int main(int argc, char** argv) {
    int   MAX_EPOCHS = (argc > 1) ? atoi(argv[1])         : 100;
    float TARGET_PCT = (argc > 2) ? (float)atof(argv[2])  : 90.0f;

    printf("HDC training from scratch with target-accuracy early stopping\n");
    printf("  Using professor's projection matrix from variables.txt\n");
    printf("  Pure float — no quantization/packing optimizations\n");
    printf("  max_epochs=%d  target_accuracy=%.1f%% (on validation)\n\n",
           MAX_EPOCHS, TARGET_PCT);

    vector<uint8_t> train_imgs, test_imgs;
    vector<uint8_t> train_lbls, test_lbls;
    int tr_n, tl_n, te_n, tel_n;
    load_mnist_images("mnist_data/train-images.idx3-ubyte", train_imgs, tr_n);
    load_mnist_labels("mnist_data/train-labels.idx1-ubyte", train_lbls, tl_n);
    load_mnist_images("mnist_data/t10k-images.idx3-ubyte",  test_imgs,  te_n);
    load_mnist_labels("mnist_data/t10k-labels.idx1-ubyte",  test_lbls, tel_n);
    printf("Loaded %d train images, %d test images from mnist_data/\n", tr_n, te_n);

    int train_size = 0, val_size = 0;
    for (int i = 0; i < tr_n; i++) {
        if (i % 5 == 0) val_size++;
        else            train_size++;
    }
    printf("Split (deterministic, every 5th -> val): %d train, %d validation, %d test (official MNIST test)\n\n",
           train_size, val_size, te_n);

    printf("Precomputing encodings for %d train + %d validation images...\n",
           train_size, val_size);
    vector<vector<float>> tr_enc(train_size);
    vector<int>            tr_y(train_size);
    vector<vector<float>> val_enc(val_size);
    vector<int>            val_y(val_size);

    int ti = 0, vi = 0;
    for (int i = 0; i < tr_n; i++) {
        if (i % 5 == 0) {
            val_enc[vi] = encode(train_imgs, i);
            normalize_f(val_enc[vi]);
            val_y[vi]   = train_lbls[i];
            vi++;
        } else {
            tr_enc[ti] = encode(train_imgs, i);
            normalize_f(tr_enc[ti]);
            tr_y[ti]   = train_lbls[i];
            ti++;
        }
    }
    printf("done\n\n");

    vector<vector<float>> proto(NUM_CLASSES, vector<float>(HD_DIM, 0.0f));
    for (int i = 0; i < train_size; i++) {
        int c = tr_y[i];
        for (int j = 0; j < HD_DIM; j++) proto[c][j] += tr_enc[i][j];
    }
    for (int c = 0; c < NUM_CLASSES; c++) normalize_f(proto[c]);

    int tr_start  = evaluate(tr_enc,  tr_y,  proto);
    int val_start = evaluate(val_enc, val_y, proto);
    printf("Training with early stopping (target=%.1f%% on validation)\n", TARGET_PCT);
    printf("Max epochs: %d\n\n", MAX_EPOCHS);
    printf("epoch  train_acc            val_acc\n");
    printf("%4d  %5d/%d = %5.1f%%   %5d/%d = %5.1f%%  (initial bundled)\n",
           0, tr_start, train_size, 100.0f * tr_start / train_size,
           val_start, val_size, 100.0f * val_start / val_size);

    int stopped_epoch = -1;
    string stop_reason;
    vector<int> order(train_size);
    for (int i = 0; i < train_size; i++) order[i] = i;

    for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
        int offset = (epoch * 481) % train_size;
        int updates = retrain_one_epoch(tr_enc, tr_y, order, offset, proto);

        int tr_correct  = evaluate(tr_enc,  tr_y,  proto);
        int val_correct = evaluate(val_enc, val_y, proto);
        string reason = check_early_stop(val_correct, val_size, TARGET_PCT);

        printf("%4d  %5d/%d = %5.1f%%   %5d/%d = %5.1f%%  (updates=%d)",
               epoch, tr_correct, train_size, 100.0f * tr_correct / train_size,
               val_correct, val_size, 100.0f * val_correct / val_size, updates);

        if (!reason.empty()) {
            printf("  * TARGET REACHED\n");
            stopped_epoch = epoch;
            stop_reason = reason;
            break;
        }
        printf("\n");
    }

    printf("\n");
    if (stopped_epoch < 0) {
        stopped_epoch = MAX_EPOCHS;
        stop_reason = "reached max_epochs (target not reached on validation)";
    }
    printf("EARLY STOP @ epoch %d: %s\n\n", stopped_epoch, stop_reason.c_str());

    printf("Final evaluation on official %d-image MNIST test set...\n", te_n);
    printf("Precomputing test encodings...\n");
    vector<vector<float>> te_enc(te_n);
    vector<int>            te_y(te_n);
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
           stopped_epoch, total_correct, te_n, ours_pct);
    printf("Delta: %+.1f percentage points\n", ours_pct - 70.7f);
    printf("-----------------------------------------------\n");

    return 0;
}
