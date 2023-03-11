architectures = dict()

kpfcn_backbone = [
    'simple',   #0
    'resnetb',
    'resnetb_strided',  #2
    'resnetb',
    'resnetb',
    'resnetb_strided',   #5
    'resnetb',
    'resnetb',
    'resnetb_strided',    #8
    'resnetb',
    'resnetb',
    'nearest_upsample',   #11
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary'
]

architectures['3dmatch'] = kpfcn_backbone
architectures['4dmatch'] = kpfcn_backbone
