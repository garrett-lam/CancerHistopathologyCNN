# CancerHistopathologyCNN

### Dataset:
This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.

The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the `Augmentor` package.

There are five classes in the dataset, each with 5,000 images, being:
- Lung benign tissue (non-cancerous)
- Lung adenocarcinoma (cancerous; organs/glands)
- Lung squamous cell carcinoma (cancerous: squamous epithelium)
- Colon benign tissue (non-cancerous)
- Colon adenocarcinoma (cancerous; organs/glands)


Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019 https://arxiv.org/abs/1912.12142v1