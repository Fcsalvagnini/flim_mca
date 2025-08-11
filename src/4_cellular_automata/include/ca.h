#ifndef _IFT_CA_
#define _IFT_CA_

#include "ift.h"

typedef struct iftCAModel {
  int i_max;
	iftMImage *orig;
  iftImage *saliency;
  iftImage *brain_mask;
  iftImage *label;
  iftFImage *fg_strength;
  iftFImage *bg_strength;
  iftAdjRel *neighborhood;
  double fg_dist, bg_dist;
} iftCAModel;

iftCAModel *InitializeCA(
  char *input_image_path, char *saliency_path, float neighborhood_radius,
  int mode, int brain
);
double evolve_state(
  int *epoch, iftMImage *orig, iftFImage **strength, iftImage **label,
  iftAdjRel *adj_rel, int label_evolving, int braain
);

float similarity_func(
  iftMImage *orig, iftImage *label, int p, int q, int label_evolving, int gray
);
iftFImage *GetProbMap(iftCAModel *ca_model);
iftImage *ExtractParasites(iftImage *img);
iftImage *ExtractTumor(iftImage *img, iftImage *brain_mask);
void evolve_ca_model(iftCAModel *ca_model, double dist, int brain);
void DestroyCAModel(iftCAModel **ca_model);
float GetFScore(iftImage *pred, iftImage *gt);

void FilterSaliencyEgg(iftImage *saliency);

// ----------------------------------------------------------------------------
typedef struct iftCASaliencyModel {
  int n_prob_maps;
  iftFImage **prob_maps;
  float *thresholds;
} iftCASaliencyModel;

iftCASaliencyModel *iftCreateCASaliencyModel(int n_layers);
void iftDestroyCASaliencyModel(iftCASaliencyModel **ca_model);
iftImage *iftIntegrateSaliencyMaps(
  char *output_dir, char *img_basename, iftCASaliencyModel *ca_model
);
float iftComputeCASigma(iftCASaliencyModel *ca_model, int p, int m, int N);
float iftCAToLogOdds(float value);

#endif