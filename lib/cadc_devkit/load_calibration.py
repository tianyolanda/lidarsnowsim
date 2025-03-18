import yaml

def load_calibration(calib_path):

  calib = {'extrinsics': yaml.load(open(calib_path + '/extrinsics.yaml'), yaml.SafeLoader),
           'CAM00': yaml.load(open(calib_path + '/00.yaml'), yaml.SafeLoader),
           'CAM01': yaml.load(open(calib_path + '/01.yaml'), yaml.SafeLoader),
           'CAM02': yaml.load(open(calib_path + '/02.yaml'), yaml.SafeLoader),
           'CAM03': yaml.load(open(calib_path + '/03.yaml'), yaml.SafeLoader),
           'CAM04': yaml.load(open(calib_path + '/04.yaml'), yaml.SafeLoader),
           'CAM05': yaml.load(open(calib_path + '/05.yaml'), yaml.SafeLoader),
           'CAM06': yaml.load(open(calib_path + '/06.yaml'), yaml.SafeLoader),
           'CAM07': yaml.load(open(calib_path + '/07.yaml'), yaml.SafeLoader)}

  return calib
