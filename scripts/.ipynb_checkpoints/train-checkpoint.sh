# #expid: 0.a
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention.yaml \
#     SESSION 0.a
    
# #expid: 0.b
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention.yaml \
#     SESSION 0.b

# #expid: 0.c # change from utils deterministic to original method in coattention
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention.yaml \
#     SESSION 0.c

# #expid: 0.d # use deterministic 123 as coattention
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention.yaml \
#     SESSION 0.d


# 1 compare adding noise at different layers
# #expid: 1.a
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0,1,2]" \
#     SESSION 1.a-v1e3_n1e2_012
    
# #expid: 1.b
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     SESSION 1.b-v1e3_n1e2_0
    
# #expid: 1.c
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[1]" \
#     SESSION 1.c-v1e3_n1e2_1
    
# #expid: 1.d
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[2]" \
#     SESSION 1.d-v1e3_n1e2_2
    
    
# #expid: 2.a
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     TRAIN.SAMPLE 1    \
#     SESSION 2.a-v1e3_n1e2_0_0.5
    
# #expid: 3.a
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     SESSION 3.a-v1e2_n1e2_0
    
# #expid: 3.b
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-4 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     SESSION 3.b-v1e4_n1e2_0
    
# #expid: 3.c
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-5 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     SESSION 3.c-v1e5_n1e2_0
  
  
# # try different noise type
# #expid: 4.a
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     MODEL.SELFATTENTION.NOISE_TYPE "blurry" \
#     SESSION 4.a_v1e3_n1e2_0_blurry


# #expid: 4.b  # to be run
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     MODEL.SELFATTENTION.NOISE_TYPE "adaptive" \
#     SESSION 4.b_v1e3_n1e2_0_adaptive

    
# #expid: 5.a
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     SESSION 5.a-v1e3_n1e3_0
    
# #expid: 5.b  # zekrom
# python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-4 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     SESSION 5.b-v1e3_n1e4_0

#expid: 6.a
python -W ignore::UserWarning tools/train.py --cfg configs/selfattention_noise.yaml \
    MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-4 \
    MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
    MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[1]" \
    SESSION 6.a-v1e4_n1e2_1
    