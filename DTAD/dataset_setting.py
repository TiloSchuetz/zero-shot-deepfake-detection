
TestDatasets = {
#TODO this doesn't work for DDIM inversion and final evaluation
    'ForenSynths': {
        'dataset_name': 'ForenSynths',
        'classes': ['progan/*', 'gaugan', 'biggan', 'cyclegan/*', 'stargan', 'stylegan/*', 'stylegan2/*']
    },
    'ForenSynths_2': {
            'dataset_name': 'ForenSynths',
            'classes':['progan/*', 'gaugan', 'biggan', 'cyclegan/*', 'stargan', 'stylegan/*']
        },
    'ForenSynths_3': {
            'dataset_name': 'ForenSynths',
            'classes':['stylegan2/*']
        },
    'Imagenet_val_5k':{
        'dataset_name': 'Imagenet_val_5k',
        'classes': ['']
    },
    'GenImage':{
        'dataset_name': 'GenImage',
        'classes':['stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'Midjourney', 'wukong', 'ADM', 'VQDM',
                 'Glide']
    },
    'New-Generator': {
        'dataset_name': 'New-Generator',
        'classes': ['flux', 'sd3', 'sdxl', 'dalle3', 'firefly', 'midjourney-v5']
    },
    'New-Generator_COCO2017_unbiased': {
        'dataset_name': 'New-Generator_COCO17_unbiased',
        'classes': ['flux', 'sd3', 'sdxl']
    },
    'New-Generator_RAISE1k_unbiased': {
        'dataset_name': 'New-Generator_RAISE1k_unbiased',
        'classes': ['dalle3', 'firefly', 'midjourney-v5']
    }   
}
