import os
from typing import Union, List, Dict, Tuple

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

from custom_utils import get_logger, parser, load_data, unet_model, dice_loss

if __name__ == '__main__':
    
    logger = get_logger(name=__name__)
    
    args = parser()
    
    # ----------------------------- Load data ----------------------------- #
    
    train_images, train_masks, val_images, val_masks = load_data({'train': args.train, 'val': args.val})
    
    # ----------------------------- Build model ----------------------------- #
    
    cnn_model = unet_model(
        image_size=(256, 256),
        aug_params={
            'random_contrast_factor': args.random_contrast_factor,
            'random_flip_mode': args.random_flip_mode,
            'random_rotation_factor': args.random_rotation_factor,
            'random_zoom_factor': args.random_zoom_factor
        },
        entry_block_filters=32,
        entry_block_kernel_size=3,
        entry_block_strides=2,
        entry_block_batch_norm_momentum=args.entry_block_batch_norm_momentum,
        down_sample_strides=(2, 2, 2),
        down_sample_kernel_sizes=(
            args.down_sample_kernel_size_0, 
            args.down_sample_kernel_size_1, 
            args.down_sample_kernel_size_2
        ),
        down_sample_batch_norm_momentums=(
            args.down_sample_batch_norm_momentum_0, 
            args.down_sample_batch_norm_momentum_1, 
            args.down_sample_batch_norm_momentum_2
        ),
        down_sample_pool_sizes=(
            args.down_sample_pool_size_0, 
            args.down_sample_pool_size_1, 
            args.down_sample_pool_size_2
        ),
        up_sample_strides=(2, 2, 2, 2),
        up_sample_kernel_sizes=(
            args.up_sample_kernel_size_0, 
            args.up_sample_kernel_size_1, 
            args.up_sample_kernel_size_2,
            args.up_sample_kernel_size_3,
        ),
        up_sample_batch_norm_momentums=(
            args.up_sample_batch_norm_momentum_0, 
            args.up_sample_batch_norm_momentum_1, 
            args.up_sample_batch_norm_momentum_2,
            args.up_sample_batch_norm_momentum_3
        ),
        up_sample_size=2,
        output_kernel_size=1,
        num_channels=1
    )
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm),
        loss=dice_loss,
        metrics=[tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True
    )
    
    cnn_model.fit(
        x=train_images,
        y=train_masks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(val_images, val_masks),
        callbacks=[early_stopping]
    )
    
    logger.info(f'Best validation dice loss: {early_stopping.best}')

    # Save model, a version number is needed for the TF serving container to load the model
    cnn_model.save(os.path.join(args.model_dir, '00000000'))