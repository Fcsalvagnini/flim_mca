// Created by Azael Souza on 22/06/21.
// Updated by Alexandre Xavier Falcao on 10/11/22

#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if ((argc != 8)&&(argc != 9))
        iftError("Usage: iftFLIM-ExtractFeaturesFromLayer P1 P2 P3 P4 P5 P6 P7 P8(optional)\n"
                  "P1: input  FLIM network architecture with a single layer (.json)\n"
                  "P2: input  folder with activation or original images (.mimg, .nii.gz or .png)\n"
                  "P3: input  list of images for feature extraction (.csv)\n"
                  "P4: input  folder with the FLIM network parameters\n"
                  "P5: input  layer index to load weights\n"
                  "P6: output folder with the resulting image features\n"
                  "P7: GPU device (0,1,..) or CPU when IFT_GPU=0\n"
                  "P8 (optional): input folder with object masks \n",
                  "main");

    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *activ_dir        = argv[2];
    char *image_list       = argv[3];
    char *param_dir        = argv[4];
    int layer_index        = atoi(argv[5]);
    char *feat_dir         = argv[6];
    int   device           = atoi(argv[7]);
    char *object_dir       = NULL;

    if (argc == 9)
        object_dir = argv[8];

    iftMakeDir(feat_dir);

    iftFLIMExtractFeaturesFromLayer(activ_dir, image_list, arch, param_dir, layer_index, feat_dir, object_dir, device);
    iftDestroyFLIMArch(&arch);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

}
