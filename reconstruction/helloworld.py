import torch
print("hello world!")
from subprocess import call
call(['convert', '-delay', '50', './t-SNE_result/3dplot*', './t-SNE_result/3dplot_anim_' + '.gif'])