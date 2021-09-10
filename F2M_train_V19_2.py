# -*- coding:utf-8 -*-
from F2M_model_V19 import *

from random import shuffle, random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256, 
                           
                           "load_size": 276,

                           "tar_size": 256,

                           "tar_load_size": 276,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,

                           "epoch_decay": 100,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/Generation/Morph/train_BM.txt",
                           
                           "A_img_path": "/yuwhan/yuwhan/Dataset/[1]Third_dataset/Morph/All/Crop_dlib/",
                           
                           "B_txt_path": "/yuwhan/yuwhan/Dataset/[2]Fourth_dataset/Generation/Morph/train_WM.txt",
                           
                           "B_img_path": "/yuwhan/yuwhan/Dataset/[1]Third_dataset/Morph/All/Crop_dlib/",

                           "age_range": [40, 64],

                           "n_classes": 48,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/48",
                           
                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/4th_paper/F2M_model_V19/checkpoint_add_en_attention",
                           
                           "sample_images": "/yuwhan/Edisk/yuwhan/Edisk/4th_paper/F2M_model_V19/sample_images_add_en_attention",
                           
                           "A_test_txt_path": "D:/[1]DB/[5]4th_paper_DB/Generation/Morph/test_BM.txt",
                           
                           "A_test_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/Crop_dlib/",
                           
                           "B_test_txt_path": "D:/[1]DB/[5]4th_paper_DB/Generation/Morph/test_WM.txt",
                           
                           "B_test_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/Crop_dlib/",
                           
                           "test_dir": "A2B",
                           
                           "fake_B_path": "C:/Users/Yuhwan/Pictures/img4",
                           
                           "fake_A_path": ""})

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
A_dataset = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)
B_dataset = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)
len_dataset = min(len(A_dataset), len(B_dataset))
G_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.epoch_decay * len_dataset)
D_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.epoch_decay * len_dataset)
g_optim = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(D_lr_scheduler, beta_1=0.5)

def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data[0])
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.tar_load_size, FLAGS.tar_load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.tar_size, FLAGS.tar_size, 3])
    B_img = B_img / 127.5 - 1.

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    B_lab = int(B_data[1])
    A_lab = int(A_data[1])

    return A_img, A_lab, B_img, B_lab

def ref_input_map(input_list):
    img = tf.io.read_file(input_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3])
    img = tf.image.per_image_standardization(img)

    return img

def generate_ref_img(input):

    ref_generator = tf.data.Dataset.from_tensor_slices(input)
    ref_generator = ref_generator.map(ref_input_map)
    ref_generator = ref_generator.batch(1)
    ref_generator = ref_generator.prefetch(tf.data.experimental.AUTOTUNE)

    ref_it = iter(ref_generator)
    ref_img = 0.
    for i in range(len(input)):
        img = next(ref_it)
        ref_img += (img / tf.reduce_max(img, [0,1,2,3])) + tf.reduce_mean(img, [0,1,2,3]) + 0.2
    ref_img = ref_img / len(input)
    ref_img = tf.clip_by_value(ref_img, -1., 1.)

    return ref_img

@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def increase_func(x):
    x = tf.cast(tf.maximum(1, x), tf.float32)
    return tf.math.log(x)

def cal_loss(A2B_model, B2A_model, DA_model, DB_model, DA_age_model, DB_age_model,
             A_batch_images, B_batch_images, B_batch_labels, A_batch_labels,
             A_ref, B_ref, A_N_buf, B_N_buf):
    
    with tf.GradientTape() as g_tape, tf.GradientTape(persistent=True) as d_tape:

        
        A_batch_label = A_batch_labels[0].numpy() - 16
        B_batch_label = B_batch_labels[0].numpy() - 16

        A_class_weights = A_N_buf[A_batch_label]
        B_class_weights = B_N_buf[B_batch_label]

        en_A_ref = tf.image.resize(A_ref, [64, 64])
        de_B_ref = tf.image.resize(B_ref, [64, 64])

        fake_B = model_out(A2B_model, [A_batch_images, B_ref, en_A_ref], True)
        fake_A_ = model_out(B2A_model, [fake_B, A_ref, de_B_ref], True)

        fake_A = model_out(B2A_model, [B_batch_images, A_ref, de_B_ref], True)
        fake_B_ = model_out(A2B_model, [fake_A, B_ref, en_A_ref], True)

        DB_real = model_out(DB_model, B_batch_images, True)
        DB_fake = model_out(DB_model, fake_B, True)
        DA_real = model_out(DA_model, A_batch_images, True)
        DA_fake = model_out(DA_model, fake_A, True)

        DB_age_real = model_out(DB_age_model, B_batch_images, True)
        DB_age_fake = model_out(DB_age_model, fake_B, True)
        DA_age_real = model_out(DA_age_model, A_batch_images, True)
        DA_age_fake = model_out(DA_age_model, fake_A, True)

        ################################################################################################
        # 나이에 대한 distance를 구하는곳
        age_loss = 0.
        for i in range(FLAGS.batch_size):
            energy_ft = (DB_age_real[i] - DB_age_fake[i])**2
            energy_ft2 = (DA_age_real[i] - DA_age_fake[i])**2

            realB_fakeB_loss = B_class_weights * tf.reduce_sum(energy_ft, -1)
            realA_fakeA_loss = A_class_weights * tf.reduce_sum(energy_ft2, -1)

            age_loss += realB_fakeB_loss + realA_fakeA_loss

        age_loss = age_loss / FLAGS.batch_size
        ################################################################################################
        # content loss 를 작성하자
        f_B = fake_B
        f_B_x, f_B_y = tf.image.image_gradients(f_B)
        f_B_m = tf.add(tf.abs(f_B_x), tf.abs(f_B_y))
        f_B = tf.abs(f_B - f_B_m)

        f_A = fake_A
        f_A_x, f_A_y = tf.image.image_gradients(f_A)
        f_A_m = tf.add(tf.abs(f_A_x), tf.abs(f_A_y))
        f_A = tf.abs(f_A - f_A_m)

        r_A = A_batch_images
        r_A_x, r_A_y = tf.image.image_gradients(r_A)
        r_A_m = tf.add(tf.abs(r_A_x), tf.abs(r_A_y))
        r_A = tf.abs(r_A - r_A_m)

        r_B = B_batch_images
        r_B_x, r_B_y = tf.image.image_gradients(r_B)
        r_B_m = tf.add(tf.abs(r_B_x), tf.abs(r_B_y))
        r_B = tf.abs(r_B - r_B_m)

        id_loss = B_class_weights * tf.reduce_mean(tf.abs(f_B - r_B)) * 5.0 \
            + A_class_weights * tf.reduce_mean(tf.abs(f_A - r_A) * 5.0)   # content loss  # style이 아닌 skin  등등 이라고 가정
        # target될 영상을 만드는것이기에, 원본의 영상이 target으로 변할 때 배경 성분은 유지 되도록

        Cycle_loss = A_class_weights * (tf.reduce_mean(tf.abs(fake_A_ - A_batch_images))) \
            * 5.0 + B_class_weights * (tf.reduce_mean(tf.abs(fake_B_ - B_batch_images))) * 5.0
        # Cycle을 하여 원본으로 갈때, Cycle된 이미지의 high freguency 성분이 원본 성분과 비슷해지도록

        G_gan_loss = B_class_weights * (tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2) \
            + A_class_weights * tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2))

        Adver_loss = B_class_weights*(tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2. \
            + A_class_weights*(tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + id_loss
        d_loss = Adver_loss + age_loss

    age_trainables_params = DA_age_model.trainable_variables + DB_age_model.trainable_variables
    g_trainables_params = A2B_model.trainable_variables + B2A_model.trainable_variables
    d_trainables_params = DA_model.trainable_variables + DB_model.trainable_variables
    g_grads = g_tape.gradient(g_loss, g_trainables_params)
    d_grads = d_tape.gradient(d_loss, d_trainables_params)
    age_grads = d_tape.gradient(d_loss, age_trainables_params)

    g_optim.apply_gradients(zip(g_grads, g_trainables_params))
    d_optim.apply_gradients(zip(d_grads, d_trainables_params))
    d_optim.apply_gradients(zip(age_grads, age_trainables_params))

    return g_loss, d_loss, age_loss


def main():
    pre_trained_encoder1 = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    pre_trained_encoder2 = tf.keras.applications.VGG16(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    A2B_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                              de_attention_shape=(FLAGS.img_size, FLAGS.img_size, 1))
    B2A_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                              de_attention_shape=(FLAGS.img_size, FLAGS.img_size, 1))
    DA_model = F2M_discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    DB_model = F2M_discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    DA_age_model = F2M_discriminator_age(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))  # 이 부분은 metric loss를 달면된다
    DB_age_model = F2M_discriminator_age(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_model=A2B_model, B2A_model=B2A_model,
                                   DA_model=DA_model,
                                   DB_model=DB_model,
                                   DA_age_model=DA_age_model,
                                   DB_age_model=DB_age_model,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    else:
        A2B_model.get_layer("conv_en_1").set_weights(pre_trained_encoder1.get_layer("conv1_conv").get_weights())
        B2A_model.get_layer("conv_en_1").set_weights(pre_trained_encoder1.get_layer("conv1_conv").get_weights())
    
        A2B_model.get_layer("conv_en_3").set_weights(pre_trained_encoder2.get_layer("block2_conv1").get_weights())
        B2A_model.get_layer("conv_en_3").set_weights(pre_trained_encoder2.get_layer("block2_conv1").get_weights())

        A2B_model.get_layer("conv_en_5").set_weights(pre_trained_encoder2.get_layer("block3_conv1").get_weights())
        B2A_model.get_layer("conv_en_5").set_weights(pre_trained_encoder2.get_layer("block3_conv1").get_weights())

    A2B_model.summary()
    DA_model.summary()
    DA_age_model.summary()

    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        A_N_list = Counter(A_labels)    # dict  class imbalance
        A_N_buf = [A_N_list[i+16] for i in range(FLAGS.n_classes)]
        A_N_buf = np.array(A_N_buf, dtype=np.float32)
        A_N_buf = (np.max(A_N_buf / FLAGS.n_classes) + 1 - (A_N_buf / FLAGS.n_classes))

        B_N_list = Counter(B_labels)    # dict  class imbalance
        B_N_buf = [B_N_list[i+16] for i in range(FLAGS.n_classes)]
        B_N_buf = np.array(B_N_buf, dtype=np.float32)
        B_N_buf = (np.max(B_N_buf / FLAGS.n_classes) + 1 - (B_N_buf / FLAGS.n_classes))

        for epoch in range(FLAGS.epochs):

            A_ref_img = generate_ref_img(A_images)  # decoder attention map
            A_ref = tf.reduce_mean(A_ref_img, -1, keepdims=True)
            A_ref = 1 / (1 + tf.exp(-4.6*A_ref))
            B_ref_img = generate_ref_img(B_images)  # decoder attention map
            B_ref = tf.reduce_mean(B_ref_img, -1, keepdims=True)
            B_ref = 1 / (1 + tf.exp(-4.6*B_ref))

            min_ = min(len(A_images), len(B_images))
            A = list(zip(A_images, A_labels))
            B = list(zip(B_images, B_labels))
            shuffle(B)
            shuffle(A)
            b_images, b_labels = zip(*B)
            a_images, a_labels = zip(*A)
            a_images = a_images[:min_]
            a_labels = a_labels[:min_]
            b_images = b_images[:min_]
            b_labels = b_labels[:min_]

            A_zip = np.array(list(zip(a_images, a_labels)))
            B_zip = np.array(list(zip(b_images, b_labels)))

            # 가까운 나이에 대해서 distance를 구하는 loss를 구성하면, 결국에는 해당이미지의 나이를 그대로 생성하는 효과?를 볼수있을것
            gener = tf.data.Dataset.from_tensor_slices((A_zip, B_zip))
            gener = gener.shuffle(len(b_images))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = min_ // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, A_batch_labels, B_batch_images, B_batch_labels = next(train_it)

                g_loss, d_loss, age_loss = cal_loss(A2B_model, B2A_model, DA_model, DB_model, DA_age_model, DB_age_model,
                                          A_batch_images, B_batch_images, B_batch_labels, A_batch_labels,
                                          A_ref, B_ref, A_N_buf, B_N_buf)
                
                print("Epoch = {}[{}/{}];\nStep(iteration) = {}\nG_Loss = {}, D_loss = {}".format(epoch,step,train_idx,
                                                                                                  count+1,
                                                                                                  g_loss, d_loss))

                if count % 100 == 0:
                    en_A_ref = tf.image.resize(A_ref, [64, 64])
                    de_B_ref = tf.image.resize(B_ref, [64, 64])

                    fake_B = model_out(A2B_model, [A_batch_images, B_ref, en_A_ref], False)
                    fake_A = model_out(B2A_model, [B_batch_images, A_ref, de_B_ref], False)

                    plt.imsave(FLAGS.sample_images + "/fake_B_{}.jpg".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/fake_A_{}.jpg".format(count), fake_A[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_B_{}.jpg".format(count), B_batch_images[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_A_{}.jpg".format(count), A_batch_images[0].numpy() * 0.5 + 0.5)


                if count % 1000 == 0:
                    num_ = int(count // 1000)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    if not os.path.isdir(model_dir):
                        print("Make {} folder to store the weight!".format(num_))
                        os.makedirs(model_dir)
                    ckpt = tf.train.Checkpoint(A2B_model=A2B_model, B2A_model=B2A_model,
                                               DA_model=DA_model,
                                               DB_model=DB_model,
                                               DA_age_model=DA_age_model,
                                               DB_age_model=DB_age_model,
                                               g_optim=g_optim, d_optim=d_optim)
                    ckpt_dir = model_dir + "/F2M_V8_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1



if __name__ == "__main__":
    main()
