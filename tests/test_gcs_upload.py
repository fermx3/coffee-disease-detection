"""
Tests para la funcionalidad de subida a GCS
"""

from unittest.mock import patch, MagicMock
import pytest
import os
from coffeedd.ml_logic.gcs_upload import (
    upload_latest_model_to_gcs,
    list_models_in_gcs,
    _verify_gcs_config,
    _find_latest_model,
)


class TestGCSUpload:

    def test_dry_run_mode(self):
        """Test que dry_run funciona sin conectar a GCS"""

        # Mock para simular que existe un modelo
        with patch("coffeedd.ml_logic.gcs_upload._find_latest_model") as mock_find:
            with patch(
                "coffeedd.ml_logic.gcs_upload._verify_gcs_config"
            ) as mock_verify:
                with patch(
                    "coffeedd.ml_logic.gcs_upload._collect_model_metadata"
                ) as mock_metadata:
                    with patch(
                        "coffeedd.ml_logic.gcs_upload._get_file_size"
                    ) as mock_size:

                        # Simular modelo encontrado (como Path o string)
                        mock_find.return_value = "/fake/path/model.h5"
                        mock_verify.return_value = True
                        mock_metadata.return_value = {
                            "file_name": "model.h5",
                            "file_size_mb": 15.5,
                            "model_type": "test",
                            "upload_timestamp": "2024-01-01T12:00:00",
                        }
                        mock_size.return_value = 15.5  # Mock del tamaño

                        # Ejecutar dry run
                        result = upload_latest_model_to_gcs(dry_run=True)

                        # Verificaciones
                        assert result["success"] == True
                        assert result["dry_run"] == True
                        assert "model_version" in result
                        assert result["local_path"] == "/fake/path/model.h5"
                        assert result["model_size_mb"] == 15.5

                        # Verificar que se llamaron los mocks correctamente
                        mock_find.assert_called_once()
                        mock_verify.assert_called_once()
                        mock_metadata.assert_called_once_with("/fake/path/model.h5")
                        mock_size.assert_called_once()

    def test_config_verification_missing_vars(self):
        """Test verificación de configuración con variables faltantes"""

        with patch.dict(os.environ, {}, clear=True):
            # Todas las variables de GCS deberían estar vacías
            result = _verify_gcs_config()
            assert result == False

    def test_config_verification_complete(self):
        """Test verificación de configuración completa"""

        test_env = {"GCP_PROJECT": "test-project", "BUCKET_NAME": "test-bucket"}

        with patch.dict(os.environ, test_env):
            with patch("google.cloud.storage.Client") as mock_client:
                # Mock del cliente GCS
                mock_bucket = MagicMock()
                mock_bucket.exists.return_value = True
                mock_client.return_value.bucket.return_value = mock_bucket

                result = _verify_gcs_config()
                assert result == True

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_find_latest_model(self, mock_glob, mock_exists):
        """Test búsqueda del último modelo"""

        # Simular que el directorio existe
        mock_exists.return_value = True

        # Simular archivos de modelo encontrados
        mock_file1 = MagicMock()
        mock_file1.stat.return_value.st_mtime = 1000
        mock_file2 = MagicMock()
        mock_file2.stat.return_value.st_mtime = 2000  # Más reciente

        mock_glob.return_value = [mock_file1, mock_file2]

        result = _find_latest_model()

        # Debería retornar el archivo más reciente
        assert result == mock_file2

    def test_model_not_found(self):
        """Test cuando no se encuentra ningún modelo"""

        with patch("coffeedd.ml_logic.gcs_upload._find_latest_model") as mock_find:
            with patch(
                "coffeedd.ml_logic.gcs_upload._verify_gcs_config"
            ) as mock_verify:

                mock_find.return_value = None
                mock_verify.return_value = True

                # Debería lanzar FileNotFoundError
                with pytest.raises(FileNotFoundError):
                    upload_latest_model_to_gcs()

    def test_debug_config_verification(self):
        """Test de debug para ver qué está pasando"""

        print("\n🔍 DEBUG: Variables de entorno actuales:")
        gcs_vars = ["GCP_PROJECT", "BUCKET_NAME", "GOOGLE_APPLICATION_CREDENTIALS"]

        for var in gcs_vars:
            value = os.environ.get(var, "NOT_SET")
            print(f"   {var}: {value}")

        with patch.dict(os.environ, {}, clear=True):
            print("\n🔍 DEBUG: Después de limpiar entorno:")
            for var in gcs_vars:
                value = os.environ.get(var, "NOT_SET")
                print(f"   {var}: {value}")

            result = _verify_gcs_config()
            print(f"\n🔍 DEBUG: _verify_gcs_config() retornó: {result}")

            assert result == False


class TestGCSIntegration:
    """Tests que requieren configuración real de GCS (opcionales)"""

    @pytest.mark.skipif(
        not all(
            [
                os.environ.get("GCP_PROJECT"),
                os.environ.get("BUCKET_NAME"),
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            ]
        ),
        reason="GCS credentials not configured",
    )
    def test_real_gcs_connection(self):
        """Test real de conexión a GCS (solo si está configurado)"""

        result = _verify_gcs_config()
        assert result == True

    @pytest.mark.skipif(
        not all(
            [
                os.environ.get("GCP_PROJECT"),
                os.environ.get("BUCKET_NAME"),
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            ]
        ),
        reason="GCS credentials not configured",
    )
    def test_list_models_real(self):
        """Test real de listado de modelos (solo si está configurado)"""

        try:
            models = list_models_in_gcs(limit=5)
            # No debería fallar, aunque la lista puede estar vacía
            assert isinstance(models, list)
        except Exception as e:
            pytest.fail(f"Error listing models: {e}")


def test_manual_upload():
    """
    Test manual para ejecutar con configuración real
    Ejecutar con: pytest tests/test_gcs_upload.py::test_manual_upload -s
    """

    if not all([os.environ.get("GCP_PROJECT"), os.environ.get("BUCKET_NAME")]):
        pytest.skip("Configuración de GCS no disponible")

    print("\n🧪 Test manual de subida a GCS")
    print("=" * 50)

    try:
        # Test de dry run
        print("1. Probando dry run...")
        result = upload_latest_model_to_gcs(dry_run=True)
        print(f"   ✅ Dry run exitoso: {result['success']}")

        # Test de listado
        print("\n2. Probando listado de modelos...")
        models = list_models_in_gcs(limit=3)
        print(f"   ✅ Encontrados {len(models)} modelos")

        # Solo probar dry run en tests automatizados
        print("\n3. Test manual completado (solo dry run en pytest)")

        print("\n✅ Test manual completado")

    except Exception as e:
        print(f"\n❌ Error en test manual: {e}")
        raise


if __name__ == "__main__":
    # Ejecutar test manual
    test_manual_upload()
