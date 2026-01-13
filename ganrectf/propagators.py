import tensorflow as tf
from ganrectf.tfutils import tfrotate


class TomoRadon:

    def __init__(self, rec, ang):
        self.rec = rec
        self.ang = ang

    def compute(self):
        nang = self.ang.shape[0]
        img = tf.transpose(self.rec, [3, 1, 2, 0])
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfrotate(img, -self.ang, interpolation="bilinear")
        sino = tf.reduce_mean(img, 1, name=None)
        sino = tf.transpose(sino, [2, 0, 1])
        sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
        return sino

    
class TomoRadon1:
    """This is similar to the Radon transformation implemented in the TomoRadon Class
    Only difference being how angle zero is interepreted 
    For this class 0 degrees is along the path of the X-ray 
    which is different from skimage Radon transformation and the TomoRadon Class"""

    def __init__(self, rec, ang, reduce_mode = 'mean'):
        self.rec = rec
        self.ang = ang
        self.mode = reduce_mode
        
    def compute(self):
        nang = self.ang.shape[0]

        # Deals with one channel at a time, a work around to operate on multi channel images
        channels = tf.split(self.rec, num_or_size_splits=self.rec.shape[-1], axis=-1)
        sinos = []
        for chn in channels:
            img = tf.transpose(chn, [3, 1, 2, 0])
            img = tf.tile(img, [nang, 1, 1, 1])
            img = tfrotate(img, self.ang, interpolation="bilinear")
            if self.mode == 'mean':
                sino = tf.reduce_mean(img, 2, name=None)
            elif self.mode == 'sum':
                sino = tf.reduce_sum(img, 2, name=None)
            sino = tf.transpose(sino, [2, 0, 1])
            sinos.append(tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1]))
        return tf.concat(sinos, -1)


class TomoFluoroLIX:
    """Fluoroscence Tomography where Absorption Correction is pre computed"""

    def __init__(self, rec, ang,  
                 Ain = None,
                 Pout = None,
                 reduce_mode = 'mean'):
        
        self.rec = rec
        self.ang = ang
        self.mode = reduce_mode
        self.Ain = Ain
        self.Pout = Pout

    def compute(self):
        nang = self.ang.shape[0]
        img = tf.transpose(self.rec, [3, 1, 2, 0])
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfrotate(img, self.ang, center = self.cen, interpolation="bilinear")

        if self.Ain is not None:
            img = tf.math.multiply(img, self.Ain)
        
        if self.Pout is not None:
            img = tf.math.multiply(img, self.Pout)

        if self.mode == 'mean':
            sino = tf.reduce_mean(img, 2, name=None)
        elif self.mode == 'sum':
            sino = tf.reduce_sum(img, 2, name=None)
        sino = tf.transpose(sino, [2, 0, 1])
        sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
        return sino

    
class TomoFluoroHXN:
    """Fluoroscence Tomography with on the fly Absorption Corrections"""
    def __init__(self, rec, ang, incident_map, emission_map, 
                 emission_kernel,
                 pix = 1., 
                 reduce_mode = 'mean'):
        self.rec = rec      # Concetrations Reconstructions i.e., the output of the NN, shape: [B, H, W, C]
        self.inc = incident_map # attenuation coefficients for incident energy, shape: [B, H, W, C]
        self.ems = emission_map     # attenuation coefficients for emission energy, shape: [B, H, W, C]
        self.ang = ang      # Array of angles, shape: [nang, 1] or [nang,]
        self.pix = pix      # Float, pixel size in cms
        # Kernel corresponds to the weights inside the cone of emission, [kH ~= 2*H, kW, C_in = 1, Cout = 1]
        self.krnl = emission_kernel     
        self.reduce_mode = reduce_mode


    def compute(self):
        
        nang = self.ang.shape[0]

        ######## Incident Attenuation ########
        # We will start with the incident attenuation
        inc = tf.transpose(self.inc, [3,1,2,0])      # [C=1, H, W, B=1]
        inc = tf.tile(inc, [nang,1,1,1])            # [Nang, H, W, B=1]
        inc = tfrotate(inc, self.ang, interpolation = "bilinear")      # [Nang, H, W, B=1]
        inc = tf.transpose(inc, [3,0,1,2])      # [B=1, Nang, H, W]

        inc_mudl = inc*self.pix          # Incident mu.dl per pixel for every angle

        # This gives you the summation of mu.dl until that pixel along the incident ray
        tau_inc = tf.cumsum(inc_mudl, axis = -1, exclusive=True)     # sum(mu.dl), [B = 1, Nang, H, W]
        tau_inc = tf.clip_by_value(tau_inc, 0.0, 200.0)      #Numerical stability

        A_in = tf.exp(-tau_inc)    # exp(-sum(mu.dl))  [B = 1, Nang, H, W]

        # consistency so A_in and P_out shapes stay the same
        A_in = A_in[0,...,None]     # [Nang, H, W, C = 1], 


        ######## Emission Attenuation ########
        ems = tf.transpose(self.ems, [3,1,2,0])      # [C=1, H, W, B=1]
        ems = tf.tile(ems, [nang,1,1,1])             # [Nang, H, W, B=1]
        ems = tfrotate(ems, self.ang, interpolation = "bilinear")      # [Nang, H, W, B=1]
        ems = tf.transpose(ems, [3,0,1,2])      # [B=1, Nang, H, W]

        ems_mudl = ems*self.pix          # Incident mu.dl per pixel for every angle

        taus = []
        for i in range(nang):
            taus.append(tf.nn.conv2d(ems_mudl[0, i:i+1,...,None],   # One angle slice at a time
                             self.krnl, 
                             strides=1, padding='SAME'))
        
        tau_ems = tf.concat(taus, axis = 0)     # [Nang, H, W, C = 1]
        tau_ems = tf.clip_by_value(tau_ems, 0.0, 200.0)      #Numerical stability
        P_out = tf.exp(-tau_ems)
        
        # Rotating Reconstruction
        img = tf.transpose(self.rec, [3, 1, 2, 0])
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfrotate(img, self.ang, interpolation="bilinear")

        # Incidnet attenuation
        img = tf.math.multiply(img, A_in)
        
        # Emission beam attenuation
        img = tf.math.multiply(img, P_out)

        if self.reduce_mode == 'mean':
            sino = tf.reduce_mean(img, 2, name=None)
        elif self.reduce_mode == 'sum':
            sino = tf.reduce_sum(img, 2, name=None)
        sino = tf.transpose(sino, [2, 0, 1])
        sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])

        return sino


class TensorRadon:

    def __init__(self, rec, ang, psi):
        self.strain_tensor = rec
        self.ang = ang
        self.psi = psi

    def tfnor_data(self, img):
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    def compute(self):
        detector_rows = self.strain_tensor.shape[0]
        detector_columns = self.strain_tensor.shape[1]
        strain_tensor = tf.cast(self.strain_tensor, dtype=tf.float32)
        vol_mask = tf.zeros((detector_rows, detector_columns, detector_columns))
        vol_mask = tf.reduce_sum(tf.abs(strain_tensor), axis=3) > 0.0
        vol_mask = tf.reshape(vol_mask, (-1, detector_columns, detector_columns, 1))
        vol_mask = tf.cast(vol_mask, dtype=tf.float32)
        angles = tf.cast(self.ang, dtype=tf.float32)
        thickness = TomoRadon(vol_mask, angles).compute()
        thickness = tf.squeeze(thickness)
        strain_tensor = tf.transpose(strain_tensor, [3, 1, 2, 0])
        proj_strain_comp = TomoRadon(strain_tensor, angles).compute()
        proj_strain_comp = tf.squeeze(proj_strain_comp)
        cos_squared = tf.expand_dims(tf.pow(tf.cos(angles), 2), 1)
        sin_squared = tf.expand_dims(tf.pow(tf.sin(angles), 2), 1)
        cos_psi_squared = tf.pow(tf.cos(self.psi), 2)
        sin_psi_squared = tf.pow(tf.sin(self.psi), 2)
        sin_2angles = tf.expand_dims(tf.sin(2 * angles), 1)
        sin_angles_sin_2psi = tf.expand_dims(tf.sin(angles) * tf.sin(2 * self.psi), 1)
        cos_angles_sin_2psi = tf.expand_dims(tf.cos(angles) * tf.sin(2 * self.psi), 1)
        if proj_strain_comp.shape[0] == 6:
            proj_strain_ws = (
                tf.multiply(proj_strain_comp[0], cos_squared * sin_psi_squared)
                + tf.multiply(proj_strain_comp[1], sin_squared * sin_psi_squared)
                + tf.multiply(proj_strain_comp[2], cos_psi_squared)
                + tf.multiply(proj_strain_comp[3], sin_2angles * sin_psi_squared)
                + tf.multiply(proj_strain_comp[4], sin_angles_sin_2psi)
                + tf.multiply(proj_strain_comp[5], cos_angles_sin_2psi)
                )
        elif proj_strain_comp.shape[0] == 3:
            proj_strain_ws = (
                tf.multiply(proj_strain_comp[0], cos_squared * sin_psi_squared)
                + tf.multiply(proj_strain_comp[1], sin_squared * sin_psi_squared)
                # + tf.multiply(proj_strain_comp[2], cos_psi_squared)
                + tf.multiply(proj_strain_comp[2], sin_2angles * sin_psi_squared)
                # + tf.multiply(proj_strain_comp[4], sin_angles_sin_2psi)
                # + tf.multiply(proj_strain_comp[5], cos_angles_sin_2psi)
                )
            
        # print(f'thickness shape is {thickness.shape}')
        # print(f'proj_strain_ws shape is {proj_strain_ws.shape}')
        # tensor_sino = tf.where(thickness > 0.05, tf.math.divide_no_nan(self.tfnor_data(proj_strain_ws), thickness), 0)
        tensor_sino = proj_strain_ws
        # tensor_sino = tf.math.divide_no_nan(proj_strain_ws, thickness)
        tensor_sino = tf.reshape(tensor_sino, [1, tensor_sino.shape[0], tensor_sino.shape[1], 1])
        return tensor_sino


class PhaseFresnel:

    def __init__(self, phase, absorption, ff, px):
        self.phase = phase
        self.absorption = absorption
        self.ff = ff
        self.px = px

    def compute(self):
        paddings = tf.constant([[self.px // 2, self.px // 2], [self.px // 2, self.px // 2]])
        pvalue = tf.reduce_mean(self.phase[:100, :])
        self.phase = tf.pad(self.phase, paddings, "SYMMETRIC")
        self.absorption = tf.pad(self.absorption, paddings, "SYMMETRIC")
        abfs = tf.complex(-self.absorption, self.phase)
        abfs = tf.exp(abfs)
        ifp = tf.abs(tf.signal.ifft2d(self.ff * tf.signal.fft2d(abfs))) ** 2
        ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
        ifp = tf.image.central_crop(ifp, 0.5)
        ifp = tf.image.per_image_standardization(ifp)
        ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
        return ifp


class PhaseFraunhofer:

    def __init__(self, phase, absorption, shift_factor=100000):
        self.phase = phase
        self.absorption = absorption
        self.shift_factor = shift_factor

    def compute(self):
        wf = tf.complex(self.absorption, self.phase)
        ifp = tf.square(tf.abs(tf.signal.fft2d(wf)))
        ifp = tf.math.log(ifp + self.shift_factor)
        ifp = tf.signal.fftshift(ifp)
        ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
        ifp = tf.image.per_image_standardization(ifp)
        # ifp = self.tfnor_diff(ifp)
        return ifp
