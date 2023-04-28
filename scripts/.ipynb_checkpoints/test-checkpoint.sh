# #expid: 1.a
# python -W ignore::UserWarning tools/test.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0,1,2]" \
#     TEST.RESUME 1.a-v1e3_n1e2_012 \
#     TEST.DATASET "unseen" \
#     TEST.MODEL "seen" \
#     TEST.EPOCH 85
    
# #expid: 1.b
# python -W ignore::UserWarning tools/test.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[0]" \
#     TEST.RESUME 1.b-v1e3_n1e2_0 \
#     TEST.DATASET "unseen" \
#     TEST.MODEL "seen" \
#     TEST.EPOCH 63
    

# #expid: 1.c
# python -W ignore::UserWarning tools/test.py --cfg configs/selfattention_noise.yaml \
#     MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
#     MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
#     MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[1]" \
#     TEST.RESUME 1.c-v1e3_n1e2_1 \
#     TEST.DATASET "unseen" \
#     TEST.MODEL "seen" \
#     TEST.EPOCH 65

#expid: 1.d
python -W ignore::UserWarning tools/test.py --cfg configs/selfattention_noise.yaml \
    MODEL.SELFATTENTION.VERB_BASE_NOISE 1e-3 \
    MODEL.SELFATTENTION.NOUN_BASE_NOISE 1e-2 \
    MODEL.SELFATTENTION.ADD_NOISE_LAYERS "[2]" \
    TEST.RESUME 1.d-v1e3_n1e2_2 \
    TEST.DATASET "unseen" \
    TEST.MODEL "seen" \
    TEST.EPOCH 41