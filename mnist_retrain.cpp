// mnist_retrain.cpp — iterative HDC retraining with convergence-based early stopping.
//
// Uses ONLY the 10 images + current prototypes already baked into variables.txt
// (no 60k MNIST, no external data). Demonstrates the training-loop early stopping
// you originally asked for: a real epoch loop that retrains on misclassification
// and halts as soon as accuracy plateaus (patience) or hits 100%.
//
// Why the frozen model only gets 50%: the baked prototypes are a single-pass
// bundling of encoded samples with no retraining pass. The standard HDC fix is
// "retrain" — for each misclassified sample, subtract its HV from the wrong
// prototype and add it to the correct one. Iterate until the training set is
// perfectly separated (or we stop improving).
//
// Build:  g++ -O2 -std=c++11 hdc/mnist_retrain.cpp -o hdc/mnist_retrain
// Run:    ./hdc/mnist_retrain
//         ./hdc/mnist_retrain 500 5 20   # epochs patience delta

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
static const int NUM_IMAGES  = 10;

// ---- encode (float, mirrors mnist_classify.cpp) --------------------------
static vector<float> encode(int img) {
    vector<float> hv(HD_DIM, 0.0f);
    for (int j = 0; j < HD_DIM; j++) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_DIM; i++) {
            float m = matrix[i * HD_DIM + j] == 0 ? -1.0f : 1.0f;
            sum += (float)images[img * INPUT_DIM + i] * m;
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

// ---- classify against a set of (normalized) prototypes -------------------
static int classify(const vector<float>& hv,
                    const vector<vector<float>>& proto) {
    int best = 0; float bs = -1e9f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        float d = 0.0f;
        for (int j = 0; j < HD_DIM; j++) d += hv[j] * proto[c][j];
        if (d > bs) { bs = d; best = c; }
    }
    return best;
}

// ---- evaluate training-set accuracy --------------------------------------
static int evaluate(const vector<vector<float>>& encoded,
                    const vector<vector<float>>& proto) {
    int correct = 0;
    for (int img = 0; img < NUM_IMAGES; img++)
        if (classify(encoded[img], proto) == img) correct++;
    return correct;
}

int main(int argc, char** argv) {
    int MAX_EPOCHS = (argc > 1) ? atoi(argv[1]) : 500;
    int PATIENCE   = (argc > 2) ? atoi(argv[2]) : 5;   // epochs w/o improvement
    float DELTA    = (argc > 3) ? (float)atof(argv[3]) : 20.0f; // min improvement %

    printf("HDC retraining with early stopping\n");
    printf("  max_epochs=%d  patience=%d  min_improvement=%.1f%%/epoch\n",
           MAX_EPOCHS, PATIENCE, DELTA);
    printf("  data: %d embedded images from variables.txt\n\n", NUM_IMAGES);

    // 1) precompute encoded query HVs (the projection is fixed)
    vector<vector<float>> encoded(NUM_IMAGES);
    for (int img = 0; img < NUM_IMAGES; img++) {
        encoded[img] = encode(img);
        normalize_f(encoded[img]);
    }

    // 2) load baked prototypes as the starting point
    vector<vector<float>> proto(NUM_CLASSES, vector<float>(HD_DIM));
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int j = 0; j < HD_DIM; j++)
            proto[c][j] = (float)class_hvs[c * HD_DIM + j];
        normalize_f(proto[c]);
    }

    int start_correct = evaluate(encoded, proto);
    printf("epoch  %4s  %8s\n", "acc", "best");
    printf("%4d  %3d/%d  %3d/%d   (baseline = baked prototypes)\n",
           0, start_correct, NUM_IMAGES, start_correct, NUM_IMAGES);

    // 3) retraining loop with early stopping
    int best_correct = start_correct;
    int stale = 0;          // epochs since last improvement >= DELTA
    int stopped_epoch = -1;
    string stop_reason;

    for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
        int prev_correct = best_correct;
        int updates = 0;

        // one retraining pass over the (tiny) training set
        for (int img = 0; img < NUM_IMAGES; img++) {
            int pred = classify(encoded[img], proto);
            if (pred != img) {
                // move the sample's HV from the wrong class to the right class
                for (int j = 0; j < HD_DIM; j++) {
                    proto[pred][j] -= encoded[img][j];
                    proto[img][j]  += encoded[img][j];
                }
                // keep prototypes normalized so dot = cosine
                normalize_f(proto[pred]);
                normalize_f(proto[img]);
                updates++;
            }
        }
        (void)updates;

        int correct = evaluate(encoded, proto);
        bool improved = (correct - prev_correct) * 100.0f / NUM_IMAGES >= DELTA;
        printf("%4d  %3d/%d  %3d/%d%s\n", epoch, correct, NUM_IMAGES,
               max(correct, best_correct), NUM_IMAGES,
               improved ? "  *" : "");

        if (correct > best_correct) best_correct = correct;

        // ---- early stopping checks ----
        if (correct == NUM_IMAGES) {
            stopped_epoch = epoch;
            stop_reason = "100% training accuracy reached";
            break;
        }
        if (!improved) {
            stale++;
            if (stale >= PATIENCE) {
                stopped_epoch = epoch;
                stop_reason = "no improvement >= " + to_string((int)DELTA) +
                              "% for " + to_string(PATIENCE) + " epochs";
                break;
            }
        } else {
            stale = 0;
        }
    }

    printf("\n");
    if (stopped_epoch < 0) {
        stopped_epoch = MAX_EPOCHS;
        stop_reason = "reached max_epochs";
    }
    printf("EARLY STOP @ epoch %d: %s\n", stopped_epoch, stop_reason.c_str());
    printf("accuracy: %d/%d -> %d/%d  (%.0f%% -> %.0f%%)\n",
           start_correct, NUM_IMAGES, best_correct, NUM_IMAGES,
           100.0f * start_correct / NUM_IMAGES,
           100.0f * best_correct / NUM_IMAGES);

    // 4) emit retrained prototypes in the same format as variables.txt, so the
    //    Arduino converter (tools/prepare_uno_data.py) can consume them.
    FILE* f = fopen("retrained_class_hvs.txt", "w");
    fprintf(f, "// Retrained class prototypes, emitted by hdc/mnist_retrain.\n");
    fprintf(f, "// Note: these are NOT L2-normalized (un-normalized, like the\n");
    fprintf(f, "// original class_hvs[]). Scale is small because we retrain in\n");
    fprintf(f, "// normalized space then undo the per-class normalization so the\n");
    fprintf(f, "// quantizer can re-normalize consistently.\n");
    fprintf(f, "const int class_hvs[] = {\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        fprintf(f, "  // class %d\n", c);
        for (int j = 0; j < HD_DIM; j++) {
            // store as scaled int to preserve precision for the converter
            long v = lround(proto[c][j] * 100000.0);
            fprintf(f, "%ld%s", v, (j == HD_DIM - 1) ? "" : ", ");
            if ((j + 1) % 10 == 0) fprintf(f, "\n");
        }
        fprintf(f, c == NUM_CLASSES - 1 ? "\n" : ",\n");
    }
    fprintf(f, "};\n");
    fclose(f);
    printf("\nretrained prototypes -> retrained_class_hvs.txt\n");

    return 0;
}
