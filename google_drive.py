"""
Google Drive integration for saving and retrieving analysis results.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path


class GoogleDriveManager:
    """
    Manage results storage to Google Drive.
    Note: Full integration requires OAuth setup. This provides the interface.
    """

    def __init__(self, folder_name='C2C12_Analysis'):
        """
        Initialize Google Drive manager.

        Parameters:
        -----------
        folder_name : str
            Name of folder in Google Drive to store results
        """
        self.folder_name = folder_name
        self.drive = None
        self.folder_id = None

    def setup_auth(self, credentials_path='credentials.json'):
        """
        Setup Google Drive authentication.

        Parameters:
        -----------
        credentials_path : str
            Path to Google OAuth credentials file

        Returns:
        --------
        bool : True if authentication successful
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build

            # For Streamlit Cloud, credentials should be in secrets
            # This is a placeholder for local testing
            if os.path.exists(credentials_path):
                credentials = Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                self.drive = build('drive', 'v3', credentials=credentials)
                return True
            else:
                print(f"Credentials file not found: {credentials_path}")
                print("For cloud deployment, add credentials to Streamlit secrets.")
                return False

        except ImportError:
            print("Google API client not installed. Using local storage only.")
            return False
        except Exception as e:
            print(f"Authentication error: {e}")
            return False

    def save_analysis_results(self, results, cell_id, output_path='./results'):
        """
        Save analysis results locally (and to Drive if authenticated).

        Parameters:
        -----------
        results : dict
            Analysis results dictionary
        cell_id : str
            Unique cell identifier
        output_path : str
            Local output directory

        Returns:
        --------
        str : Path to saved file
        """
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{cell_id}_{timestamp}.json"
        filepath = output_dir / filename

        # Add metadata
        results_with_meta = {
            'metadata': {
                'cell_id': cell_id,
                'timestamp': timestamp,
                'software': 'C2C12_Analyzer_v1',
                'model': 'Lulevich2006'
            },
            'results': results
        }

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_with_meta, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")

        # Try to upload to Drive if authenticated
        if self.drive:
            self._upload_to_drive(filepath)

        return str(filepath)

    def _upload_to_drive(self, local_path):
        """
        Upload file to Google Drive.

        Parameters:
        -----------
        local_path : str
            Path to local file

        Returns:
        --------
        bool : True if upload successful
        """
        try:
            from googleapiclient.http import MediaFileUpload

            file_name = Path(local_path).name

            # Find or create folder
            if not self.folder_id:
                self._find_or_create_folder()

            # Upload file
            file_metadata = {
                'name': file_name,
                'parents': [self.folder_id]
            }

            media = MediaFileUpload(local_path, mimetype='application/json')
            file = self.drive.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            print(f"Uploaded to Google Drive: {file.get('id')}")
            return True

        except Exception as e:
            print(f"Upload to Drive failed: {e}")
            return False

    def _find_or_create_folder(self):
        """
        Find or create analysis folder in Google Drive.

        Returns:
        --------
        str : Folder ID
        """
        try:
            # Search for folder
            results = self.drive.files().list(
                q=f"name='{self.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces='drive',
                fields='files(id, name)',
                pageSize=1
            ).execute()

            folders = results.get('files', [])

            if folders:
                self.folder_id = folders[0]['id']
            else:
                # Create new folder
                file_metadata = {
                    'name': self.folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.drive.files().create(
                    body=file_metadata,
                    fields='id'
                ).execute()
                self.folder_id = folder.get('id')

            return self.folder_id

        except Exception as e:
            print(f"Folder management error: {e}")
            return None

    def list_analyses(self):
        """
        List all saved analyses in Google Drive.

        Returns:
        --------
        list : List of file metadata
        """
        try:
            if not self.folder_id:
                self._find_or_create_folder()

            results = self.drive.files().list(
                q=f"'{self.folder_id}' in parents and trashed=false",
                spaces='drive',
                fields='files(id, name, createdTime, modifiedTime, size)',
                pageSize=100
            ).execute()

            return results.get('files', [])

        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def get_summary_markdown(self, results, cell_id):
        """
        Generate markdown summary of analysis results.

        Parameters:
        -----------
        results : dict
            Analysis results
        cell_id : str
            Cell identifier

        Returns:
        --------
        str : Markdown formatted summary
        """
        md = f"""# Cell Compression Analysis Summary

**Cell ID:** {cell_id}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Membrane Elasticity

"""
        if 'membrane' in results:
            mem = results['membrane']
            md += f"""- **Young's Modulus (Em):** {mem.get('Em_MPa', 'N/A'):.2f} MPa
- **Bending Constant (Km):** {mem.get('Km_kT', 'N/A'):.1f} kT
- **Fit R²:** {mem.get('r_squared', 'N/A'):.4f}
- **Data Points:** {mem.get('n_points', 'N/A')}
- **Fitting Range (ε):** [{mem.get('epsilon_range', [0, 0.3])[0]:.4f}, {mem.get('epsilon_range', [0, 0.3])[1]:.4f}]

"""

        if 'cytoskeleton' in results:
            cyto = results['cytoskeleton']
            md += f"""## Cytoskeleton Elasticity

- **Young's Modulus (Ei):** {cyto.get('Ei_kPa', 'N/A'):.2f} kPa
- **Fit R²:** {cyto.get('r_squared', 'N/A'):.4f}
- **Data Points:** {cyto.get('n_points', 'N/A')}
- **Fitting Range (ε):** [{cyto.get('epsilon_range', [0, 0.3])[0]:.4f}, {cyto.get('epsilon_range', [0, 0.3])[1]:.4f}]

"""

        if 'rupture' in results:
            rup = results['rupture']
            md += f"""## Rupture Analysis

- **Rupture Point (ε):** {rup.get('epsilon', 'N/A'):.4f}
- **Rupture Force:** {rup.get('force', 'N/A'):.2e} N
- **Peaks Detected:** {rup.get('n_peaks_detected', 0)}

"""

        if 'alignment' in results:
            align = results['alignment']
            md += f"""## Compression Quality

- **Alignment Score:** {align.get('mean_alignment_score', 'N/A'):.3f}
- **Head-on Compression:** {align.get('is_head_on', False)}
- **Quality Assessment:** {align.get('quality_assessment', 'Unknown')}

"""

        return md
