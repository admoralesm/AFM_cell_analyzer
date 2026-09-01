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
        Open the spreadsheet the service account has been given access to.

        A service account is not a person: it has its own Drive with a storage
        quota of zero bytes. Asking it to create a spreadsheet therefore fails
        with

            APIError [403]: The user's Drive storage quota has been exceeded

        which reads like the wrong problem entirely. The fix is never to create
        the file from here. Make the spreadsheet yourself, in your own Drive,
        share it with the service account's client_email as an Editor, and pass
        its id. Creation is still attempted as a last resort when no id is
        configured, but the quota failure is reported with what to do about it.

        Parameters
        ----------
        spreadsheet_id : str, optional
            Spreadsheet id, or a full Google Sheets URL. Falls back to
            ``st.secrets["google_sheets"]["spreadsheet_id"]``.
        """
        if not self.is_authenticated:
            return False

        spreadsheet_id = spreadsheet_id or self._configured_id()
        if spreadsheet_id:
            spreadsheet_id = self.extract_spreadsheet_id(spreadsheet_id)

        try:
            if spreadsheet_id:
                self.spreadsheet = self.client.open_by_key(spreadsheet_id)
            else:
                try:
                    # Works when a sheet of this name has been shared with the
                    # service account.
                    self.spreadsheet = self.client.open(self.sheet_name)
                except gspread.SpreadsheetNotFound:
                    self.spreadsheet = self.client.create(self.sheet_name)
                    st.info(f"Created a new Google Sheet: {self.sheet_name}")
        except Exception as exc:
            self._report_open_failure(exc, spreadsheet_id)
            return False

        try:
            try:
                self.worksheet = self.spreadsheet.worksheet("Cells")
            except gspread.WorksheetNotFound:
                self.worksheet = self.spreadsheet.add_worksheet(
                    title="Cells", rows=1000, cols=12
                )
                self._initialize_headers()
            return True
        except Exception as exc:
            st.error(f"Opened the spreadsheet but could not use it: {exc}")
            return False

    @staticmethod
    def extract_spreadsheet_id(value: str) -> str:
        """Accept either a bare id or a full Google Sheets URL."""
        import re

        value = (value or "").strip()
        match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]{20,})", value)
        return match.group(1) if match else value

    def _configured_id(self) -> str:
        """Spreadsheet id from secrets, if one was configured."""
        try:
            section = st.secrets.get("google_sheets", {})
            return section.get("spreadsheet_id") or section.get("spreadsheet_url") or ""
        except Exception:
            return ""

    def service_account_email(self) -> str:
        try:
            return st.secrets["google_sheets_credentials"].get("client_email", "")
        except Exception:
            return ""

    def _report_open_failure(self, exc: Exception, spreadsheet_id: Optional[str]):
        """Turn Google's API errors into something actionable."""
        message = str(exc)
        email = self.service_account_email() or "your service account's client_email"

        if "storage quota" in message.lower():
            st.error(
                "Google refused to create the spreadsheet because the service "
                "account has no Drive storage of its own. Service accounts always "
                "have a zero-byte quota, so this is not something you can raise."
            )
            st.info(
                "Do this instead:\n\n"
                "1. Create a blank Google Sheet in your own Drive.\n"
                f"2. Share it with **{email}** as an **Editor**.\n"
                "3. Copy its id from the URL, the long string between `/d/` and "
                "`/edit`.\n"
                "4. Paste it into the Spreadsheet ID box in the sidebar, or add it "
                "to `.streamlit/secrets.toml` as:\n\n"
                "```toml\n[google_sheets]\nspreadsheet_id = \"your-id-here\"\n```"
            )
        elif "PERMISSION_DENIED" in message or "[403]" in message:
            st.error(
                f"The service account cannot open that spreadsheet. Share it with "
                f"**{email}** as an Editor and try again."
            )
        elif "not found" in message.lower() or "[404]" in message:
            st.error(
                f"No spreadsheet with id `{spreadsheet_id}`. Check that you copied "
                f"the part of the URL between `/d/` and `/edit`."
            )
        else:
            st.error(f"Error accessing spreadsheet: {message}")

    # The columns this app writes, in the order a new sheet gets them. An
    # existing sheet keeps whatever order it already has: rows are written by
    # header name, never by position, so a column moved or renamed by hand
    # does not silently put the nucleus modulus in the notes column.
    COLUMNS = [
        ("cell_id", "Cell ID"),
        ("date_analyzed", "Date Analyzed"),
        ("cell_height", "Cell Height (μm)"),
        ("cantilever_constant", "Cantilever Constant (pN/nm)"),
        ("Em", "Young's Modulus (Em, MPa)"),
        ("Ei", "Young's Modulus (Ei, kPa)"),
        ("En", "Young's Modulus (En, kPa)"),
        ("membrane_areal", "Membrane Em·h (mN/m)"),
        ("break_1", "ε₁ membrane hands over"),
        ("break_2", "ε₂ nucleus engages"),
        ("model", "Model"),
        ("combination", "Combination"),
        ("fit_range", "Fitted range"),
        ("video_link", "Video Link"),
        ("force_curve_created", "Force Curve Created"),
        ("fit_quality", "Fit Quality (R²)"),
        ("notes", "Notes"),
        ("analysis_status", "Analysis Status"),
        ("timestamp", "Timestamp"),
    ]

    def _initialize_headers(self):
        """Put the full header row on a brand new sheet."""
        try:
            self.worksheet.insert_row([name for _, name in self.COLUMNS], 1)
        except Exception as e:
            st.warning(f"Could not insert headers: {str(e)}")

    def _header_row(self):
        """The sheet's current header, adding any columns it does not have."""
        try:
            header = self.worksheet.row_values(1)
        except Exception:
            header = []
        header = [h for h in header if str(h).strip()]
        if not header:
            self._initialize_headers()
            return [name for _, name in self.COLUMNS]

        missing = [name for _, name in self.COLUMNS if name not in header]
        if missing:
            # Extend to the right rather than inserting. Inserting would shift
            # every cell in every existing row, and this sheet may already
            # hold results someone has referenced by cell address.
            header = header + missing
            try:
                self.worksheet.update(
                    values=[header],
                    range_name=f"A1:{gspread.utils.rowcol_to_a1(1, len(header))}",
                )
            except Exception as e:
                st.warning(f"Could not add the new columns: {e}")
        return header

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

            # Build the row against the sheet's own header, so a column the
            # user has moved still receives the right value and a column the
            # app does not know about is left alone rather than overwritten.
            defaults = {
                "date_analyzed": datetime.now().strftime("%Y-%m-%d"),
                "force_curve_created": "Yes",
                "analysis_status": "Complete",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            by_name = {}
            for key, name in self.COLUMNS:
                value = cell_data.get(key, defaults.get(key, ""))
                by_name[name] = "" if value is None else value

            header = self._header_row()
            row = [by_name.get(name, "") for name in header]

            self.worksheet.append_row(row, value_input_option="USER_ENTERED")

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


def initialize_sheets_manager(
    sheet_name: str = "AFM_Cell_Database",
    spreadsheet_id: Optional[str] = None,
) -> Optional[GoogleSheetsManager]:
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
        if manager.get_or_create_sheet(spreadsheet_id):
            return manager

    return None
