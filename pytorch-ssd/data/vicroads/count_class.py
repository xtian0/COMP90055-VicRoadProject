import os

site_label = 'site4a_077'


img_dir = site_label+'/'
annotation_dir = site_label+'/'

prefix = '20191210-0957_CAM2_0077_'

output_dir = './'
output_file = site_label+'_train.txt'

last_index = 80



f = open(output_dir+output_file, "w")
instance_count = 0

dataset_size = last_index+1
for img_count in range(0, dataset_size):
    try:
        img_id = str(img_count).zfill(4)
        in_name = prefix+img_id 
        out_name = site_label+'_'+img_id

        in_img_name = img_dir+in_name+'.jpg'
        out_img_name = img_dir+out_name+'.jpg'
        print(in_img_name, out_img_name)
        os.rename(in_img_name, out_img_name)

        in_ana_name = annotation_dir+in_name+'.xml'
        out_ana_name = annotation_dir+out_name+'.xml'
        print(in_ana_name, out_ana_name)
        os.rename(in_ana_name, out_ana_name)

        f.write(out_name+'\n')
        instance_count+=1
    except:
        print("Caution! Failure on instance", img_count)
    
f.close()
print("Converted Instances Count:", instance_count)




