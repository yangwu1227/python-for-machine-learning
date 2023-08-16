from __future__ import annotations
import os
from typing import Tuple, Union, List, Dict, Any
import pickle
import boto3
import json
import logging
from functools import partial
import s3fs
from math import sqrt

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

from hydra import compose, initialize, core
from omegaconf import OmegaConf

from base_trainer import BaseTrainer

# -------------------------- Shifted patch tokenizer ------------------------- #

class ShiftedPatchTokenizer(tf.keras.layers.Layer):
    """
    This class implements the Shifted Patch Tokenization technique proposed in the paper.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_patches: int,
        projection_dim: int,
        **kwargs
        ):
        """
        Constructor for the ShiftedPatchTokenizer class.

        Parameters
        ----------
        image_size : int
            Size of the image.
        patch_size : int
            Size of the patch.
        num_patches : int
            Number of non-overlapping patches (H * W / patch_size^2).
        projection_dim : int
            Dimension of the patch embeddings.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        # Each input image is spatially shifted by half the patch size
        self.half_patch = patch_size // 2
        # Flatten the patches into sequence of vectors
        self.num_patches = num_patches
        self.flatten_patches = tf.keras.layers.Reshape((self.num_patches, -1))
        # Linearly project the flattened patches vectors into the space of hidden dimension
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=self.projection_dim, name='projection_layer')
        self.layer_norm = tf.keras.layers.LayerNormalization(name='normalization_layer')

    def shift_crop_pad(self, images: tf.Tensor, mode: str) -> tf.Tensor:
        """
        This function shifts, crops, and pads the input images.

        Parameters
        ----------
        images : tf.Tensor
            Input images with shape (batch_size, image_size, image_size, num_channels).
        mode : str
            Mode of shifting, which must be one of the following:
            - left-up
            - left-down
            - right-up
            - right-down

        Returns
        -------
        tf.Tensor
            Shifted images.

        Raises
        ------
        ValueError
            If the mode is not one of the above mentioned modes.
        """
        # Shift the batch of images
        if mode == 'left-up':
            # Up and left 
            crop_height = self.half_patch
            crop_width = self.half_patch
            # Down and right 
            shift_height = 0
            shift_width = 0
        elif mode == 'left-down':
            # Up and left
            crop_height = 0
            crop_width = self.half_patch
            # Down and right
            shift_height = self.half_patch
            shift_width = 0
        elif mode == 'right-up':
            # Up and left
            crop_height = self.half_patch
            crop_width = 0
            # Down and right
            shift_height = 0
            shift_width = self.half_patch
        elif mode == 'right-down':
            # Up and left
            crop_height = 0
            crop_width = 0
            # Down and right
            shift_height = self.half_patch
            shift_width = self.half_patch
        else:
            raise ValueError(f'Invalid mode: {mode}, must be one of the following: left-up, left-down, right-up, right-down')

        # Crop the shifted images (after cropping, the images dimensions are reduced by half the patch size)
        cropped_images = tf.image.crop_to_bounding_box(
            images,
            # Top left corner of the bounding box is at (offset_height, offset_width) = (crop_height, crop_width)
            offset_height=crop_height,
            offset_width=crop_width,
            # Lower right corner is at (offset_height + target_height, offset_width + target_width)
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch
        )

        # Pad the cropped images to restore the original image size
        padded_cropped_images = tf.image.pad_to_bounding_box(
            cropped_images,
            # Add 'offset_height' rows of zeros on top
            offset_height=shift_height,
            # Add 'offset_width' columns of zeros on the left
            offset_width=shift_width,
            # Pad the images on the bottom and right until it has dimensions (target_height, target_width)
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return padded_cropped_images

    def call(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        This function implements the call method of the ShiftedPatchTokenizer class. It 
        tokenizes the input images into patches.

        Parameters
        ----------
        images : tf.Tensor
            Input images with shape (batch_size, image_size, image_size, num_channels).

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Tuple containing the visual tokens and the patches (non-flattened).
        """
        # Concat the shifted images with the original image along the channel (depth) dimension (batch_size, image_size, image_size, 5 x 3)
        images = tf.concat(
            values=[
                # Four shifted and one original image
                images,
                self.shift_crop_pad(images=images, mode='left-up'),
                self.shift_crop_pad(images=images, mode='left-down'),
                self.shift_crop_pad(images=images, mode='right-up'),
                self.shift_crop_pad(images=images, mode='right-down')
            ],
            # The channel dimension now has 5 x 3 = 15 channels or feature maps
            axis=-1,
        )

        # Divide the concatenated features into patches (batch_size, num_patches, patch_size, patch_size, 5 x 3) where 'num_patches' = (image_size / patch_size)^2
        patches = tf.image.extract_patches(
            images=images,
            # Each patch is of size (patch_size, patch_size, 5 x 3)
            sizes=[1, self.patch_size, self.patch_size, 1],
            # Ensure non-overlapping patches by setting strides equal to the patch size
            strides=[1, self.patch_size, self.patch_size, 1],
            # Regular sampling
            rates=[1, 1, 1, 1],
            # No padding
            padding='VALID'
        )
        # Flatten each patch (batch_size, num_patches, patch_size x patch_size x 5 x 3)
        flattened_patches = self.flatten_patches(patches)
        
        # Normalization and linearly projection on last dimension of the flattened patches (patch_size x patch_size x 5 x 3)
        visual_tokens = self.layer_norm(flattened_patches)
        # Shape of visual tokens is (batch_size, num_patches, projection_dim)
        visual_tokens = self.projection(visual_tokens)

        return (visual_tokens, patches)

    def get_config(self) -> Dict[str, Any]:
        """
        This function implements the get_config method of the ShiftedPatchTokenizer class.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the configuration of the ShiftedPatchTokenizer instance.
        """
        base_config = super(ShiftedPatchTokenizer, self).get_config()
        config = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ShiftedPatchTokenizer:
        """
        This function implements the from_config method of the ShiftedPatchTokenizer class.
        This is a no-op since all the parameters that are passed to the constructor of the 
        class are already serializable.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing the configuration of the ShiftedPatchTokenizer instance.

        Returns
        -------
        ShiftedPatchTokenizer
            ShiftedPatchTokenizer instance with the specified configuration.
        """        
        return cls(**config)

# ------------------------------- Patch encoder ------------------------------ #

class PatchEncoder(tf.keras.layers.Layer):
    """
    This class implements the PatchEncoder layer.
    """
    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        """
        Constructor for the PatchEncoder class.

        Parameters
        ----------
        num_patches : int
            Number of patches (H * W / patch_size^2).
        projection_dim : int
            Dimension of the patch embeddings.
        """
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        # Map each patch index (single scalar) to a vector of size 'projection_dim'
        self.position_encoding = tf.keras.layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim,
            embeddings_initializer=tf.keras.initializers.HeNormal()
        )
        # This is a 1-D vector from 0 to 'num_patches - 1'
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, visual_tokens: tf.Tensor) -> tf.Tensor:
        """
        This function implements the call method of the PatchEncoder class. It
        encodes the visual tokens with positional embeddings by adding the same
        set of positional embeddings to the visual tokens for all the examples
        in the batch.

        Parameters
        ----------
        visual_tokens : tf.Tensor
            Encoded visual tokens.
        """
        # Broadcasts positional encodings (num_patches, projection_dim) to the shape of visual tokens (batch_size, num_patches, projection_dim)
        encoded_visual_tokens = visual_tokens + self.position_encoding(self.positions)
        return encoded_visual_tokens

    def get_config(self) -> Dict[str, Any]:
        """
        This function implements the get_config method of the PatchEncoder class.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the configuration of the PatchEncoder instance.
        """
        base_config = super(PatchEncoder, self).get_config()
        config = {
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> PatchEncoder:
        """
        This function implements the from_config method of the PatchEncoder class.
        Again, this is a no-op since all the parameters are passed to the constructor 
        of the class are already serializable.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing the configuration of the PatchEncoder instance.

        Returns
        -------
        PatchEncoder
            PatchEncoder instance with the specified configuration.
        """        
        return cls(**config)

# ------------------------- Multi-head attention LSA ------------------------- #

class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    """
    This class implements the multi-head attention layer with the locality self-attention (LSA) 
    mechanism. In the original paper, the authors used LSA to solve the lack of locality inductive
    bias in ViTs. 

    The problem:

    "The second problem is the poor attention mechanism. The feature dimension of image data is far 
    greater than that of natural language and audio signal, so the number of embedded tokens is inevitably 
    large. Thus, the distribution of attention scores of tokens becomes smooth [i.e., uniform]. In other words, 
    we face the problem that ViTs cannot attend locally to important visual tokens."
    
    Solution:
    
    "LSA mitigates the smoothing phenomenon of attention score distribution by excluding self-tokens 
    and by applying learnable temperature to the softmax function. LSA induces attention to work 
    locally by forcing each token to focus more on tokens with large relation to itself."
    """
    def __init__(self, **kwargs):
        """
        Constructor for the MultiHeadAttentionLSA class.
        """
        super().__init__(**kwargs)
        # Trainable temperature parameter initialized to sqrt(d_k) where d_k is the dimension of the key vectors
        self.temperature = tf.Variable(
            initial_value=sqrt(self._key_dim),
            trainable=True
        )

    def _compute_attention(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, attention_mask: tf.Tensor = None, training: bool = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        This function applies dot-product attention with query, key, value tensors. We keep the `call` method of the MultiHeadAttention class 
        unchanged, but this function defines the computation inside `call` with projected multi-head Q, K, V inputs.

        - B: Batch size
        - N: Number of attention heads
        - T: Number of tokens or elements in the target sequence (from the query)
        - S: Number of tokens or elements in the source sequence (from the key)
        - H: Depth or dimensionality of each attention head (also referred to as value_dim in the code)

        Parameters
        ----------
        query : tf.Tensor
            Projected query `Tensor` of shape `(B, T, N, key_dim)`.
        key : tf.Tensor
            Projected key `Tensor` of shape `(B, S, N, key_dim)`.
        value : tf.Tensor
            Projected value `Tensor` of shape `(B, S, N, value_dim)`.
        attention_mask : tf.Tensor
            Boolean mask of shape `(B, T, S)`, that prevents attention to certain positions.
        training : bool
            Whether the layer should behave in training mode (add dropout) or in inference mode. Defaults to `None`.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Multi-headed outputs of attention computation and attention weights.
        """
        # Scale the query tensor by 1 over the temperature parameter
        query = tf.multiply(x=query, y=(1.0 / self.temperature))
        # Dot product between 'query' and 'key' to get the raw attention scores
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        # Normalize the attention scores to probabilities, masking out (set to -infinity) the diagonal elements of the similarity matrix
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        # Dropping out entire tokens to attend to
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        # Output of the attention mechanism (weighted sum of the values)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)

        # Attention output has shape (B, T, N, H) and attention scores has shape (B, N, T, S)
        return attention_output, attention_scores

# ------------------------------------ MLP ----------------------------------- #

def mlp(x: tf.Tensor, hidden_units: List[int], dropout_rate: float, mode: str) -> tf.Tensor:
    """
    Multi-layer perceptron (MLP) with dropout. The original paper for the 'Vision Transformer' architecture uses two hidden layers with a 
    GELU activation function.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    hidden_units : List[int]
        Hidden units of the MLP.
    dropout_rate : float
        Dropout rate.
    mode : str
        Whether this the transformer MLP block or the MLP head, must be one of 'transformer-block' or 'head'.

    Returns
    -------
    tf.Tensor
        Output tensor of the MLP block.
    """
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu, kernel_initializer='he_normal', name=f'dense_{mode}_{i}')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f'dense_{mode}_dropout_{i}')(x)
    return x

# ------------------------------- Trainer class ------------------------------ #

class VisionTransformerTrainer(BaseTrainer):
    """
    This class performs training and evaluation of the Vision Transformer model with shifted patch tokenization and locality self-attention (LSA) based 
    on the paper 'Vision Transformer for Small-Size Datasets'.

    The three model variants in Table 1 of the Vision Transformer paper:

    +-------------+-------+------------+---------+-------+--------+
    | Model       | Layers| Project Dim| MLP size| Heads | Params |
    +-------------+-------+------------+---------+-------+--------+
    | ViT-Base    | 12    | 768        | 3072    | 12    | 86M   |
    | ViT-Large   | 24    | 1024       | 4096    | 16    | 307M  |
    | ViT-Huge    | 32    | 1280       | 5120    | 16    | 632M  |
    +-------------+-------+------------+---------+-------+--------+
    """
    def __init__(self, 
                 hyperparameters: Dict[str, Any],
                 config: Dict[str, Any],
                 job_name: str,
                 train_dataset: tf.data.Dataset,
                 val_dataset: tf.data.Dataset,
                 train_class_weights: Dict[str, float],
                 distributed: bool,
                 strategy: tf.distribute.Strategy,
                 model_dir: str,
                 logger: logging.Logger) -> None:
        """
        Constructor for the BaselineTrainer class.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            A dictionary containing the hyperparameters for model training.
        config : Dict[str, Any]
            A dictionary containing the configuration for model training.
        job_name : str
            The name of the job.
        train_dataset : tf.data.Dataset
            A tf.data.Dataset object that contains the training data.
        val_dataset : tf.data.Dataset
            The validation data is recommend to be a repeated dataset.
        train_class_weights : Dict[str, float]
            Class weights for the training data.
        distributed : bool
            A boolean that specifies whether to use distributed training.
        strategy : tf.distribute.Strategy
            A tf.distribute.Strategy object that specifies the strategy for distributed training.
        model_dir : str
            Path to the directory where the model will be saved.
        logger : logging.Logger
            A logger object.

        Returns
        -------
        None
        """
        super().__init__(
            hyperparameters=hyperparameters,
            config=config,
            job_name=job_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_class_weights=train_class_weights,
            distributed=distributed,
            strategy=strategy,
            model_dir=model_dir,
            logger=logger
        )

    def _create_model(self) -> tf.keras.Model:
        """
        Function that creates the compiled model.

        Returns
        -------
        tf.keras.Model
            The compiled model.
        """
        # Data augmentation layers
        data_augmentation = AugmentationModel(aug_params={
                'RandomFlip': {'mode': self.hyperparameters['random_flip_mode']},
                'RandomRotation': {'factor': self.hyperparameters['random_rotation_factor']},
                'RandomZoom': {'height_factor': self.hyperparameters['random_zoom_height_factor'], 'width_factor': self.hyperparameters['random_zoom_width_factor']},
                'RandomContrast': {'factor': self.hyperparameters['random_contrast_factor']}
        }).build_augmented_model()

        # ---------------------------- Model architecture ---------------------------- #

        inputs = tf.keras.Input(shape=(self.config['image_size'], self.config['image_size'], self.config['num_channels']), name='input_layer')
        x = data_augmentation(inputs)

        # Shifted patch tokenization and patch embedding layer
        num_patches = (self.config['image_size'] // self.hyperparameters['spt_patch_size']) ** 2
        (visual_tokens, patches) = ShiftedPatchTokenizer(
            image_size=self.config['image_size'],
            patch_size=self.hyperparameters['spt_patch_size'],
            num_patches=num_patches,
            projection_dim=self.hyperparameters['spt_projection_dim'],
            name='shifted_patch_tokenizer'
        )(x)
        encoded_visual_tokens = PatchEncoder(num_patches=num_patches, projection_dim=self.hyperparameters['spt_projection_dim'], name='patch_encoder')(visual_tokens)

        # Build the diagonal attention mask (0's for diagonal elements and 1's for all other elements)
        diag_attn_mask = tf.expand_dims((1 - tf.eye(num_patches, dtype=tf.int8)), axis=0)
        # Transformer encoder block
        for block in range(self.hyperparameters['transformer_num_layers']):

            # Layer normalization 1
            norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'transformer_block_{block}_layer_normalization_1')(encoded_visual_tokens)
            # Multi-head attention
            attention_output = MultiHeadAttentionLSA(
                num_heads=self.hyperparameters['transformer_num_heads'],
                key_dim=self.hyperparameters['spt_projection_dim'],
                dropout=self.hyperparameters['transformer_dropout_rate'],
                kernel_initializer='he_normal',
                name=f'transformer_block_{block}_multi_head_attention'
            )(query=norm_1, value=norm_1, key=norm_1, attention_mask=diag_attn_mask)
            # Skip connection 1
            skip_con_1 = tf.keras.layers.Add(name=f'transformer_block_{block}_skip_connection_1')([attention_output, encoded_visual_tokens])
            # Layer normalization 2
            norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'transformer_block_{block}_layer_normalization_2')(skip_con_1)
            # MLP
            norm_2 = mlp(
                x=norm_2, 
                hidden_units=[
                    self.hyperparameters['transformer_mlp_multiple_0'] * self.hyperparameters['spt_projection_dim'],
                    # The second hidden layer should project back to the 'projection_dim' dimension for the skip connection to work below
                    self.hyperparameters['spt_projection_dim']
                ],
                dropout_rate=self.hyperparameters['transformer_dropout_rate'],
                mode=f'transformer_block_{block}',
            )
            # Skip connection 2
            encoded_visual_tokens = tf.keras.layers.Add(name=f'transformer_block_{block}_skip_connection_2')([norm_2, skip_con_1])

        # Create a (batch_size, projection_dim) tensor (matrix)
        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_normalization')(encoded_visual_tokens)
        representation = tf.keras.layers.Flatten(name='flatten')(representation)
        representation = tf.keras.layers.Dropout(rate=self.hyperparameters['dense_dropout_rate'], name='dropout')(representation)

        # MLP head
        hidden_units = [self.hyperparameters[f'dense_units_{i}'] for i in range(self.hyperparameters['dense_num_layers'])]
        features = mlp(
            x=representation,
            hidden_units=hidden_units,
            dropout_rate=self.hyperparameters['dense_dropout_rate'],
            mode='head'
        )
        # Output layer
        outputs = tf.keras.layers.Dense(units=self.config['num_classes'], activation='linear', name='output_layer')(features)
        model = tf.keras.Model(inputs, outputs)

        # ---------------------------------- Compile --------------------------------- #

        optimizer = self._create_optimizer(learning_rate=self.hyperparameters['adam_initial_lr'])
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        return model

    def _count_trainable_weights(self, model: tf.keras.Model) -> int:
        """
        Function that counts the number of trainable weights in a model.

        Parameters
        ----------
        model : tf.keras.Model
            A tf.keras.Model object.

        Returns
        -------
        int
            The number of trainable weights in the model.
        """
        trainable_params_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        return trainable_params_count

    def fit(self) -> None:
        """
        Function that fits the models.

        Returns
        -------
        None
        """
        # ------------------------------- Create model ------------------------------- #

        if self.distributed:
            with self.strategy.scope():
                model = self._create_model()
        else:
            model = self._create_model()

        trainable_params_count = self._count_trainable_weights(model)
        self.logger.info(f'Number of trainable parameters for training: {trainable_params_count}')
        del trainable_params_count

        # --------------------------------- Callbacks -------------------------------- #

        # The 'on_train_begin' method resets the 'self.wait' attribute to 0 so this can be reused across multiple calls to 'fit'
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['patience'],
            mode='max',
            restore_best_weights=True
        )
        back_and_restore = tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(os.getcwd(), 'backup'),
            # Delete the backup directory after the training is completed, so the next call to 'fit' will create a new backup directory
            delete_checkpoint=True
        )
        callbacks = [early_stopping, back_and_restore]
        if self.distributed:
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=f's3://{self.config["s3_bucket"]}/{self.config["s3_key"]}/tensorboard_logs/{self.job_name}'
            )
            callbacks.append(tensorboard)

        # ---------------------------------- Fit model --------------------------------- #

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters['fit_epochs'],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # Number of steps (batches of samples) to draw from before stopping validation
            validation_steps=self.hyperparameters['fit_validation_steps'],
            class_weight=self.train_class_weights
        )

        self.logger.info(f'Best validation accuracy: {early_stopping.best}')

        # -------------------------------- Save model -------------------------------- #

        if self.distributed:
            # For single-host multi-gpu training, there is no cluster resolver so we specify the type and id manually
            if self.strategy.cluster_resolver is None:
                model_dir = self._create_model_dir(
                    self.model_dir, 
                    'worker', 
                    0
                )
            else:
                # If the cluster resolver is not None, we are in multi-host training mode
                model_dir = self._create_model_dir(
                    self.model_dir, 
                    self.strategy.cluster_resolver.task_type, 
                    self.strategy.cluster_resolver.task_id
                )
            model.save(os.path.join(model_dir, '0'))
        else:
            model.save(os.path.join(self.model_dir, '0'))

        return None

if __name__ == '__main__':

    from custom_utils import get_logger, parser, add_additional_args, load_dataset, AugmentationModel

    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger(name='vision_transformer')

    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='vision_transformer')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    additional_args = {
        # Data augmentation parameters
        'random_flip_mode': str,
        'random_rotation_factor': float,
        'random_contrast_factor': float,
        'random_zoom_height_factor': float,
        'random_zoom_width_factor': float,
        # Architecture parameters
        'spt_patch_size': int,
        'spt_projection_dim': int,
        'transformer_num_heads': int,
        'transformer_num_layers': int,
        'transformer_dropout_rate': float,
        'transformer_mlp_multiple_0': int,
        'dense_num_layers': int,
        'dense_units_0': int,
        'dense_units_1': int,
        'dense_units_2': int,
        'dense_dropout_rate': float,
        # Optimization, loss, and fit parameters
        'adam_initial_lr': float,
        'adam_beta_1': float,
        'adam_beta_2': float,
        'adam_clipnorm': float,
        'use_focal_loss': int,
        'loss_gamma': float,
        'loss_alpha': float,
        'fit_epochs': int,
        # Distributed training parameters
        'distributed_multi_worker': int
    }

    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

    job_name = args.training_env['job_name']

    # Strategy for distributed training
    if args.test_mode:
        distributed = False
        strategy = None
    else:
        distributed = True
        if args.distributed_multi_worker:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.distribute.MirroredStrategy()

    # --------------------------------- Load data -------------------------------- #

    if args.test_mode:

        train_dataset = load_dataset(
            dir=args.train,
            batch_size=config['batch_size']
        ).take(2)

        val_dataset = load_dataset(
            dir=args.val,
            batch_size=config['batch_size']
        ).take(2)

    else:
        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        global_batch_size = config['batch_size'] * num_workers

        train_dataset = load_dataset(
            dir=args.train,
            batch_size=global_batch_size
        )

        val_dataset = load_dataset(
            dir=args.val,
            batch_size=global_batch_size
        )

    fs = s3fs.S3FileSystem()
    with fs.open(f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/train_weights.json', 'rb') as f:
        train_class_weights = json.load(f)
    # Convert all keys to integers
    train_class_weights = {int(k): v for k, v in train_class_weights.items()}

    # --------------------------------- Train model --------------------------------- #

    trainer = VisionTransformerTrainer(
        hyperparameters={
            # Data augmentation parameters
            'random_flip_mode': args.random_flip_mode,
            'random_rotation_factor': args.random_rotation_factor,
            'random_contrast_factor': args.random_contrast_factor,
            'random_zoom_height_factor': args.random_zoom_height_factor,
            'random_zoom_width_factor': args.random_zoom_width_factor,
            # Architecture parameters
            'spt_patch_size': args.spt_patch_size,
            'spt_projection_dim': args.spt_projection_dim,
            'transformer_num_heads': args.transformer_num_heads,
            'transformer_num_layers': args.transformer_num_layers,
            'transformer_dropout_rate': args.transformer_dropout_rate,
            'transformer_mlp_multiple_0': args.transformer_mlp_multiple_0,
            'dense_num_layers': args.dense_num_layers,
            'dense_units_0': args.dense_units_0,
            'dense_units_1': args.dense_units_1,
            'dense_units_2': args.dense_units_2,
            'dense_dropout_rate': args.dense_dropout_rate,
            # Optimization, loss, and fit parameters
            'adam_initial_lr': args.adam_initial_lr,
            'adam_beta_1': args.adam_beta_1,
            'adam_beta_2': args.adam_beta_2,
            'adam_clipnorm': args.adam_clipnorm,
            'use_focal_loss': args.use_focal_loss,
            'loss_gamma': args.loss_gamma,
            'loss_alpha': args.loss_alpha,
            'fit_epochs': args.fit_epochs,
            'fit_validation_steps': 1 if args.test_mode else len(val_dataset),
        },
        config=config,
        job_name=job_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset.repeat(),
        train_class_weights=train_class_weights,
        distributed=distributed,
        strategy=strategy,
        model_dir=args.model_dir,
        logger=logger
    )

    trainer.fit()

    del trainer