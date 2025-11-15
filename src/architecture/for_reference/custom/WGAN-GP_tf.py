import tensorflow as tf

def gradient_penalty(self, batch_size, real_images, fake_images):
    # Each image int he batch gets a random number between 0 & 1 and stored in vector alpha
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)

    # Calculate set of interpolated images
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    # Critic scores each interpolated image
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = self.critic(interpolated, training=True)

    # Gradient of prediction calculated with respect to input image
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # Calculate L2 norm of gradient vector
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    # Return average squared distance between the L2 nrom and I
    gp = tf.redice_mean((norm - 1.0) ** 2)
    return gp

def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]

    # Perform three critic updates
    for i in range(3):
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        with tf.GradientTape() as tape:
            fake_images = self.generator(
                random_latent_vectors, training = True
            )
            fake_predictions = self.critic(fake_images, training = True)
            real_predictions = self.critic(real_images, training = True)

            # Calculate Wasserstein loss for the critic
            # the difference between the average prediction for the fake images annd real images
            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                real_predictions
            )

            # Calculate gradient penalty term
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images
            )

            # Critic loss function as a weighted sum of the Wasserstein loss and gradient penalty
            c_loss = c_wass_loss + c_gp * self.gp_weight
        
        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)

        # Update the weights of the critic
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        )

    random_latent_vectors = tf.random.normal(
        shape=(batch_size)
    )

    # Calculate the Wasserstein loss for the generator
    with tf.GradientTape() as tape:
        fake_images = self.generator(random_latent_vectors, training=True)
        fake_predictions = self.critic(fake_images, training = True)
        g_loss = -tf.reduce_mean(fake_predictions)
 
    # Update the weights of the generator
    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    )

    self.c_loss_metric.update_state(c_loss)
    self.c_wass_loss_metric.update_state(c_wass_loss)
    self.c_gp_metric.update_state(c_gp)
    self.g_loss_metric.update_state(g_loss)

    return {m.name: m.result() for m in self.metrics}
