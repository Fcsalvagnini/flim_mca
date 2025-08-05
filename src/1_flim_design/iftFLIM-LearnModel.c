// Created by Azael Souza on 22/06/21.
// Updated by Alexandre Xavier Falcao on 10/11/22

#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 5)
      iftError("Usage: iftFLIM-LearnModel P1 P2 P3 P4\n"
        "P1: input  FLIM network architecture (.json)\n"
        "P2: input  folder with the original images (.png, .nii.gz)\n"
        "P3: input  folder with the training markers (-seeds.txt)\n"
        "P4: output folder with the FLIM network parameters per layer\n",
        "main");

    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *orig_dir         = argv[2];
    char *markers_dir      = argv[3];
    char *param_dir        = argv[4];
    iftMakeDir(param_dir);

    iftFLIMLearnModel(orig_dir,markers_dir, param_dir, arch);
    iftDestroyFLIMArch(&arch);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
