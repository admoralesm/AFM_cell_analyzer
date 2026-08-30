"""
Google Sheets Database Manager for AFM Cell Analysis
Handles read/write operations to Google Sheets for persistent cell data storage
"""

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json


class GoogleSheetsManager:
    """Manage Google Sheets database for cell analysis results"""

    def __init__(self, sheet_name: str = "AFM_Cell_Database"):
        """
        Initialize Google Sheets manager

        Parameters:
        -----------
        sheet_name : str
            Name of the Google Sheet to use
        """
        self.sheet_name = sheet_name
        self.client = None
        self.spreadsheet = None
        self.worksheet = None
        self.is_authenticated = False

    def authenticate(self) -> bool:
        """
        Authenticate with Google Sheets API using Streamlit secrets

        Returns:
        --------
        bool : True if authentication successful, False otherwise
        """
        try:
            # Get credentials from Streamlit secrets
            if "google_sheets_credentials" not in st.secrets:
                st.error(
                    "❌ Google Sheets credentials not found in Streamlit secrets. "
                    "Please add 'google_sheets_credentials' to your .streamlit/secrets.toml"
                )
                return False

            # Parse credentials
            creds_dict = st.secrets["google_sheets_credentials"]

            # Authenticate with Google Sheets
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]

            credentials = Credentials.from_service_account_info(
                creds_dict,
                scopes=scopes
            )

            # Create client
            self.client = gspread.authorize(credentials)
            self.is_authenticated = True

            return True

        except Exception as e:
            st.error(f"❌ Authentication failed: {str(e)}")
            return False

    def get_or_create_sheet(self, spreadsheet_id: Optional[str] = None) -> bool:
        """
        Get existing spreadsheet or create new one

        Parameters:
        -----------
        spreadsheet_id : str, optional
            ID of existing spreadsheet. If None, will try to find by name or create new

        Returns:
        --------
        bool : True if successful
        """
        try:
            if not self.is_authenticated:
                return False

            if spreadsheet_id:
                # Open by ID
                self.spreadsheet = self.client.open_by_key(spreadsheet_id)
            else:
                # Try to open by name, or create if doesn't exist
                try:
                    self.spreadsheet = self.client.open(self.sheet_name)
                except gspread.SpreadsheetNotFound:
                    # Create new spreadsheet
                    self.spreadsheet = self.client.create(self.sheet_name)
                    st.info(f"✅ Created new Google Sheet: {self.sheet_name}")
                    print(f"Created spreadsheet: {self.spreadsheet.url}")

            # Get or create worksheet
            try:
                self.worksheet = self.spreadsheet.worksheet("Cells")
            except gspread.WorksheetNotFound:
                # Create worksheet with headers
                self.worksheet = self.spreadsheet.add_worksheet(
                    title="Cells",
                    rows=1000,
                    cols=12
                )
                self._initialize_headers()

            return True

        except Exception as e:
            st.error(f"❌ Error accessing spreadsheet: {str(e)}")
            return False

    def _initialize_headers(self):
        """Initialize worksheet with column headers"""
        headers = [
            "Cell ID",
            "Date Analyzed",
            "Cell Height (μm)",
            "Cantilever Constant (pN/nm)",
            "Young's Modulus (Em, MPa)",
            "Young's Modulus (Ei, kPa)",
            "Video Link",
            "Force Curve Created",
            "Fit Quality (R²)",
            "Notes",
            "Analysis Status",
            "Timestamp"
        ]

        try:
            self.worksheet.insert_row(headers, 1)
        except Exception as e:
            st.warning(f"Could not insert headers: {str(e)}")

    def append_cell_data(self, cell_data: Dict) -> Tuple[bool, str]:
        """
        Append a new cell analysis to the database

        Parameters:
        -----------
        cell_data : dict
            Dictionary containing:
            - cell_id (required): Cell name/ID
            - date_analyzed: Date of analysis
            - cell_height: Cell height in micrometers
            - cantilever_constant: Cantilever constant in pN/nm
            - Em: Membrane Young's modulus in MPa
            - Ei: Cytoskeleton Young's modulus in kPa
            - video_link: Google Drive video link (optional)
            - force_curve_created: Yes/No
            - fit_quality: R² value
            - notes: Analysis notes

        Returns:
        --------
        (success, message) : tuple of bool and status message
        """
        try:
            if not self.worksheet:
                return False, "Worksheet not initialized"

            # Validate required fields
            if "cell_id" not in cell_data or not cell_data["cell_id"]:
                return False, "Cell ID is required"

            # Prepare row data
            row = [
                cell_data.get("cell_id", ""),
                cell_data.get("date_analyzed", datetime.now().strftime("%Y-%m-%d")),
                cell_data.get("cell_height", ""),
                cell_data.get("cantilever_constant", ""),
                cell_data.get("Em", ""),
                cell_data.get("Ei", ""),
                cell_data.get("video_link", ""),
                cell_data.get("force_curve_created", "No"),
                cell_data.get("fit_quality", ""),
                cell_data.get("notes", ""),
                cell_data.get("analysis_status", "Complete"),
                datetime.now().isoformat()
            ]

            # Append to worksheet
            self.worksheet.append_row(row)

            return True, f"✅ Cell {cell_data['cell_id']} saved to database"

        except Exception as e:
            return False, f"❌ Error appending data: {str(e)}"

    def get_all_cells(self) -> pd.DataFrame:
        """
        Retrieve all cell data from database

        Returns:
        --------
        DataFrame : All cell records
        """
        try:
            if not self.worksheet:
                return pd.DataFrame()

            # Get all values
            all_values = self.worksheet.get_all_values()

            if len(all_values) <= 1:
                return pd.DataFrame()

            # Convert to DataFrame
            headers = all_values[0]
            data = all_values[1:]

            df = pd.DataFrame(data, columns=headers)

            # Convert numeric columns
            numeric_cols = [
                "Cell Height (μm)",
                "Cantilever Constant (pN/nm)",
                "Young's Modulus (Em, MPa)",
                "Young's Modulus (Ei, kPa)",
                "Fit Quality (R²)"
            ]

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            st.error(f"❌ Error retrieving data: {str(e)}")
            return pd.DataFrame()

    def search_cells(self, search_term: str) -> pd.DataFrame:
        """
        Search cells by ID

        Parameters:
        -----------
        search_term : str
            Cell ID or partial match to search for

        Returns:
        --------
        DataFrame : Matching cells
        """
        df = self.get_all_cells()

        if df.empty:
            return df

        # Search in Cell ID column (case-insensitive)
        mask = df["Cell ID"].astype(str).str.contains(
            search_term,
            case=False,
            na=False
        )

        return df[mask]

    def filter_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter cells by analysis date range

        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns:
        --------
        DataFrame : Cells within date range
        """
        df = self.get_all_cells()

        if df.empty or "Date Analyzed" not in df.columns:
            return df

        try:
            df["Date Analyzed"] = pd.to_datetime(df["Date Analyzed"])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            mask = (df["Date Analyzed"] >= start) & (df["Date Analyzed"] <= end)
            return df[mask]

        except Exception as e:
            st.error(f"❌ Error filtering by date: {str(e)}")
            return df

    def sort_by_modulus(self, descending: bool = True) -> pd.DataFrame:
        """
        Sort cells by Young's modulus

        Parameters:
        -----------
        descending : bool
            If True, sort from highest to lowest

        Returns:
        --------
        DataFrame : Sorted cells
        """
        df = self.get_all_cells()

        if df.empty or "Young's Modulus (Em, MPa)" not in df.columns:
            return df

        df_sorted = df.sort_values(
            "Young's Modulus (Em, MPa)",
            ascending=not descending,
            na_position='last'
        )

        return df_sorted

    def delete_cell(self, cell_id: str) -> Tuple[bool, str]:
        """
        Delete a cell record from database

        Parameters:
        -----------
        cell_id : str
            Cell ID to delete

        Returns:
        --------
        (success, message) : tuple of bool and status message
        """
        try:
            if not self.worksheet:
                return False, "Worksheet not initialized"

            # Get all values
            all_values = self.worksheet.get_all_values()

            # Find row with matching cell_id
            row_to_delete = None
            for i, row in enumerate(all_values, 1):
                if i > 1 and len(row) > 0 and row[0] == cell_id:
                    row_to_delete = i
                    break

            if row_to_delete is None:
                return False, f"Cell {cell_id} not found"

            # Delete row
            self.worksheet.delete_rows(row_to_delete)

            return True, f"✅ Cell {cell_id} deleted from database"

        except Exception as e:
            return False, f"❌ Error deleting cell: {str(e)}"

    def export_to_csv(self) -> Optional[str]:
        """
        Export database to CSV format

        Returns:
        --------
        str : CSV data as string, or None if error
        """
        try:
            df = self.get_all_cells()

            if df.empty:
                return None

            return df.to_csv(index=False)

        except Exception as e:
            st.error(f"❌ Error exporting to CSV: {str(e)}")
            return None

    def export_to_json(self) -> Optional[str]:
        """
        Export database to JSON format

        Returns:
        --------
        str : JSON data as string, or None if error
        """
        try:
            df = self.get_all_cells()

            if df.empty:
                return None

            # Convert to JSON with proper handling of NaN values
            return df.to_json(orient='records', date_format='iso')

        except Exception as e:
            st.error(f"❌ Error exporting to JSON: {str(e)}")
            return None

    def get_statistics(self) -> Dict:
        """
        Calculate database statistics

        Returns:
        --------
        dict : Statistics about analyzed cells
        """
        df = self.get_all_cells()

        if df.empty:
            return {
                "total_cells": 0,
                "avg_em": 0,
                "avg_ei": 0,
                "avg_fit_quality": 0
            }

        stats = {
            "total_cells": len(df),
            "avg_em": pd.to_numeric(df["Young's Modulus (Em, MPa)"], errors='coerce').mean(),
            "avg_ei": pd.to_numeric(df["Young's Modulus (Ei, kPa)"], errors='coerce').mean(),
            "avg_fit_quality": pd.to_numeric(df["Fit Quality (R²)"], errors='coerce').mean(),
            "cells_with_video": len(df[df["Video Link"].astype(str).str.len() > 0]),
            "cells_with_force_curve": len(df[df["Force Curve Created"] == "Yes"])
        }

        return stats

    def get_spreadsheet_url(self) -> str:
        """
        Get the URL of the Google Sheet

        Returns:
        --------
        str : URL to the spreadsheet
        """
        if self.spreadsheet:
            return self.spreadsheet.url
        return ""


def initialize_sheets_manager(sheet_name: str = "AFM_Cell_Database") -> Optional[GoogleSheetsManager]:
    """
    Initialize and authenticate Google Sheets manager

    Parameters:
    -----------
    sheet_name : str
        Name of the Google Sheet to use

    Returns:
    --------
    GoogleSheetsManager or None if authentication fails
    """
    manager = GoogleSheetsManager(sheet_name)

    if manager.authenticate():
        if manager.get_or_create_sheet():
            return manager

    return None
