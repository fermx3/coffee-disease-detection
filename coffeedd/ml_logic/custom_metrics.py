import tensorflow as tf
import keras


class DiseaseRecallMetric(keras.metrics.Metric):
    """
    Metrica personalizada para calcular el recall específico para clases de enfermedades,
    excluyendo la clase 'healthy'.
    """

    def __init__(self, name="disease_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_disease_samples = self.add_weight(
            name="total_disease_samples", initializer="zeros"
        )
        self.correct_disease_predictions = self.add_weight(
            name="correct_disease_predictions", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertir one-hot a índices
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)

        # Máscara para todas las enfermedades (no healthy = índice 0)
        disease_mask = tf.not_equal(y_true_idx, 0)

        # Contar muestras de enfermedad
        disease_samples = tf.reduce_sum(tf.cast(disease_mask, tf.float32))

        # Contar predicciones correctas de enfermedad
        correct_disease = tf.logical_and(disease_mask, tf.equal(y_true_idx, y_pred_idx))
        correct_disease_count = tf.reduce_sum(tf.cast(correct_disease, tf.float32))

        self.total_disease_samples.assign_add(disease_samples)
        self.correct_disease_predictions.assign_add(correct_disease_count)

    def result(self):
        return tf.math.divide_no_nan(
            self.correct_disease_predictions, self.total_disease_samples
        )

    def reset_state(self):
        self.total_disease_samples.assign(0.0)
        self.correct_disease_predictions.assign(0.0)

    def get_config(self):
        """Método necesario para serialización"""
        config = super().get_config()
        return config
