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
            # Use the tab that is already there. Creating a "Cells" tab when
            # the spreadsheet has one perfectly good sheet is how rows end up
            # somewhere the user is not looking: they see their header on the
            # first tab and the app appends to a second one behind it.
            try:
                self.worksheet = self.spreadsheet.worksheet("Cells")
            except gspread.WorksheetNotFound:
                worksheets = self.spreadsheet.worksheets()
                if worksheets:
                    self.worksheet = worksheets[0]
                    if not [c for c in self.worksheet.row_values(1) if str(c).strip()]:
                        self._initialize_headers()
                else:
                    self.worksheet = self.spreadsheet.add_worksheet(
                        title="Cells", rows=1000, cols=len(self.COLUMNS)
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

    # The columns this app writes, in the order a new sheet gets them. Each
    # modulus is followed by the stretch of deformation it was measured over,
    # because a modulus without its range is not a result you can compare.
    # The video link goes last: it is the widest cell and the least often
    # read, so it belongs off the end rather than in the middle.
    COLUMNS = [
        ("cell_id", "Cell ID"),
        ("experiment_date", "Experiment Date"),
        ("cell_height", "Cell Height (μm)"),
        ("spring_constant", "Spring Constant, K (N/m)"),
        # The membrane's in-plane tension, written before the moduli because
        # it is the first spring to answer. Blank for a cell type whose
        # membrane is modelled as one spring, which is not the same as zero.
        ("T0", "Membrane tension (T₀, mN/m)"),
        ("T0_range", "T₀ range (ε)"),
        ("Em", "Young's Modulus (Em, MPa)"),
        ("Em_range", "Em range (ε)"),
        # The cortical layer, for cell types that have one. Blank where the
        # cell type does not, which is not the same as zero.
        ("Ecx", "Young's Modulus (Ecx cortex, kPa)"),
        ("Ei", "Young's Modulus (Ei, kPa)"),
        ("Ei_range", "Ei range (ε)"),
        # The nuclear envelope, likewise.
        ("Ene", "Young's Modulus (Ene envelope, MPa)"),
        ("En", "Young's Modulus (En, kPa)"),
        ("En_range", "En range (ε)"),
        # Only the four-regime C2C12 fit writes these: the cytoskeleton
        # around the nucleus (regime 3), the apparent contact modulus
        # (regime 1), and every coefficient and anchor as one JSON cell.
        ("E_nc", "Young's Modulus (E_nc perinuclear cytoskeleton, kPa)"),
        ("E_align", "Apparent contact modulus (E_align, kPa)"),
        ("piecewise", "4-regime coefficients (JSON)"),
        ("membrane_areal", "Membrane Em·h (mN/m)"),
        ("model", "Model"),
        ("combination", "Combination"),
        ("break_1", "ε₁ membrane hands over"),
        ("break_2", "ε₂ deep layer engages"),
        ("fit_range", "Fitted range (ε)"),
        ("fit_quality", "Fit Quality (R²)"),
        ("adj_r_squared", "Adjusted R²"),
        ("chi_squared", "Chi squared"),
        ("chi_squared_reduced", "Chi squared / dof"),
        ("noise_sigma", "Measured noise σ (N)"),
        ("rmse_N", "RMSE (N)"),
        ("n_points", "Points fitted"),
        ("weighting", "Weighting"),
        ("cell_radius", "Cell radius R₀ (μm)"),
        ("nucleus_radius", "Deep layer radius (μm)"),
        ("membrane_thickness", "Membrane thickness (nm)"),
        ("protein_coat", "Protein coat thickness (nm)"),
        ("sarcomere_relaxed", "Relaxed sarcomere (nm)"),
        ("sarcomere_at_max", "Sarcomere at ε_max (nm)"),
        ("poisson", "Poisson (membrane / interior)"),
        # Measured off the video frame rather than typed in, and a note
        # about what the video showed. Both belong with the summary because
        # they are what somebody looking at an odd modulus checks first.
        ("video_height_um", "Height (um)"),
        ("video_comment", "Video Comment"),
        ("force_curve_created", "Force Curve Created"),
        ("analysis_status", "Analysis Status"),
        ("notes", "Notes"),
        ("timestamp", "Timestamp"),
        ("video_link", "Video Link"),
    ]

    # The first tab holds the columns somebody reads across a population:
    # who the cell was, how stiff each material came out, over what stretch
    # of the curve, and how well the model followed it. Everything else is
    # the working, and it goes on a second tab so this one stays readable
    # on a laptop screen without scrolling sideways.
    MAIN_KEYS = (
        "experiment_date", "cell_id", "cell_height", "spring_constant",
        "Em", "Em_range", "Ecx", "Ei", "Ei_range", "Ene", "En", "En_range",
        "fit_quality", "chi_squared", "video_height_um", "video_comment",
    )
    # Repeated at the front of the second tab, so a row there can be matched
    # back to its cell without counting rows.
    EXTRA_KEYS = ("cell_id", "experiment_date", "spring_constant")
    EXTRA_TAB = "Additional info"

    @classmethod
    def main_columns(cls):
        """(key, heading) for the first tab, in the order it is written."""
        names = dict(cls.COLUMNS)
        return [(key, names[key]) for key in cls.MAIN_KEYS if key in names]

    @classmethod
    def extra_columns(cls):
        """(key, heading) for the second tab: the working, keyed by cell."""
        names = dict(cls.COLUMNS)
        keys = list(cls.EXTRA_KEYS) + [
            key for key, _ in cls.COLUMNS
            if key not in cls.MAIN_KEYS and key not in cls.EXTRA_KEYS
        ]
        return [(key, names[key]) for key in keys if key in names]

    # Columns this app used to write, and what they are called now. A header
    # is renamed in place so the data under it stays put; without this the new
    # name would look like a missing column and be appended alongside the old
    # one, leaving two half-filled columns saying the same thing.
    RENAMED = {
        # Named after a myoblast's anatomy, which is wrong for a cell that
        # has no nucleus term. The data under them is unchanged.
        "ε₂ nucleus engages": "ε₂ deep layer engages",
        "Nucleus radius (μm)": "Deep layer radius (μm)",
        "Date Analyzed": "Experiment Date",
        "Cantilever Constant (pN/nm)": "Spring Constant, K (N/m)",
        "Fitted range": "Fitted range (ε)",
        "Membrane Eₘ·h (mN/m)": "Membrane Em·h (mN/m)",
    }

    @staticmethod
    def unique_headers(names):
        """
        A header row with no two columns sharing a name, and none blank.

        A real sheet acquires duplicates: a column added by hand under a
        name the app also writes, a rename that lands on a heading already
        there, a stray copy. Pandas tolerates that; Arrow does not, and the
        database tab died with "Duplicate column names found" while showing
        nothing about which ones. Later copies are numbered rather than
        dropped, because a column with data in it is somebody's data even
        when its name is a mistake.
        """
        seen, out = {}, []
        for index, raw in enumerate(names):
            name = str(raw).strip() or f"Column {index + 1}"
            if name in seen:
                seen[name] += 1
                name = f"{name} ({seen[name]})"
                # The numbered name could itself collide with a real one.
                while name in seen:
                    seen[name] = seen.get(name, 1) + 1
                    name = f"{name} ({seen[name]})"
            seen[name] = 1
            out.append(name)
        return out

    def _initialize_headers(self, worksheet=None, columns=None):
        """Put the header row on a brand new tab."""
        worksheet = worksheet if worksheet is not None else self.worksheet
        columns = columns if columns is not None else self.main_columns()
        try:
            worksheet.insert_row([name for _, name in columns], 1)
        except Exception as e:
            st.warning(f"Could not insert headers: {str(e)}")

    def extra_worksheet(self, create=True):
        """
        The second tab, holding the working. None when there is none yet.

        Made on demand rather than up front, so a spreadsheet somebody
        already keeps does not sprout an empty tab just for being opened.
        """
        # getattr, not attribute access: a manager built for a test, or one
        # whose sheet was never opened, has no spreadsheet at all, and the
        # row on the first tab must still be written.
        if getattr(self, "spreadsheet", None) is None:
            return None
        try:
            return self.spreadsheet.worksheet(self.EXTRA_TAB)
        except gspread.WorksheetNotFound:
            if not create:
                return None
        except Exception:
            return None
        try:
            sheet = self.spreadsheet.add_worksheet(
                title=self.EXTRA_TAB, rows=1000,
                cols=max(len(self.extra_columns()), 10),
            )
        except Exception as exc:
            st.warning(f"Could not add the '{self.EXTRA_TAB}' tab: {exc}")
            return None
        self._initialize_headers(sheet, self.extra_columns())
        return sheet

    def _header_row(self, worksheet=None, columns=None):
        """
        A tab's header, brought up to date with the current layout.

        Renames first, so data already under an old name stays with it; then
        the columns this app writes but the tab lacks. Order is only changed
        when there is nothing to lose, which is why `reorder_columns` is a
        separate, deliberate call.

        A heading that belongs on the other tab is left exactly where it is.
        A sheet written before the split has every column on the first tab,
        and quietly deleting half of them to tidy it up would throw away
        somebody's data; `reorder_columns` is the deliberate move.
        """
        worksheet = worksheet if worksheet is not None else self.worksheet
        columns = columns if columns is not None else self.main_columns()
        try:
            header = [h for h in worksheet.row_values(1) if str(h).strip()]
        except Exception:
            header = []

        if not header:
            self._initialize_headers(worksheet, columns)
            return [name for _, name in columns]

        renamed = self.unique_headers(
            [self.RENAMED.get(name, name) for name in header]
        )
        missing = [name for _, name in columns if name not in renamed]
        if renamed != header or missing:
            renamed = renamed + missing
            try:
                worksheet.update(
                    values=[renamed],
                    range_name=f"A1:{gspread.utils.rowcol_to_a1(1, len(renamed))}",
                )
            except Exception as e:
                st.warning(f"Could not update the header: {e}")
                return header
        return renamed

    def reorder_columns(self):
        """
        Rewrite the sheet in this app's column order, carrying the data.

        Every row is remapped by column name, so nothing lands under the wrong
        heading, and a column the app does not know about is kept and pushed
        to the end rather than dropped. This rewrites the whole sheet, so it
        is only ever run when asked for.
        """
        try:
            values = self.worksheet.get_all_values()
        except Exception as exc:
            return False, f"Could not read the sheet: {exc}"
        if not values:
            self._initialize_headers()
            return True, "Wrote the header to an empty sheet."

        header = self.unique_headers(
            [self.RENAMED.get(h, h) for h in values[0]]
        )
        rows_by_name = [
            dict(zip(header, raw)) for raw in values[1:]
            if any(str(cell).strip() for cell in raw)
        ]

        main = [name for _, name in self.main_columns()]
        extra = [name for _, name in self.extra_columns()]
        # A heading this app has never heard of is somebody's own column. It
        # is kept, at the end of the first tab, rather than dropped.
        unknown = [h for h in header if h and h not in main and h not in extra]

        def rewrite(worksheet, wanted, source):
            body = [[row.get(name, "") for name in wanted] for row in source]
            worksheet.clear()
            worksheet.update(
                values=[wanted] + body,
                range_name=(
                    f"A1:{gspread.utils.rowcol_to_a1(len(body) + 1, len(wanted))}"
                ),
            )

        # The working tab is written first, and the summary is only stripped
        # down to the summary columns once that has actually succeeded.
        # The other order loses data: a spreadsheet where the second tab
        # cannot be made would have its working columns removed from the
        # first tab and written nowhere.
        note, moved = "", False
        sheet = self.extra_worksheet()
        if sheet is None:
            note = (f" The '{self.EXTRA_TAB}' tab could not be made, so "
                    f"every column is still on this one.")
        else:
            try:
                # Rows already on the second tab are read first, so a sheet
                # that has been split once is not split again onto itself
                # and rows written straight there are not lost.
                existing = sheet.get_all_values()
                if len(existing) > 1:
                    seen = [dict(zip([self.RENAMED.get(h, h) for h in existing[0]],
                                     raw))
                            for raw in existing[1:]
                            if any(str(cell).strip() for cell in raw)]
                else:
                    seen = []
                merged = seen if len(seen) >= len(rows_by_name) else rows_by_name
                rewrite(sheet, extra, merged)
                moved = True
                note = f" The working is on the '{self.EXTRA_TAB}' tab."
            except Exception as exc:
                note = (f" The '{self.EXTRA_TAB}' tab was not written "
                        f"({exc}), so every column is still on this one.")

        main_header = (
            main + unknown if moved
            else main + [h for h in header if h and h not in main]
        )
        try:
            rewrite(self.worksheet, main_header, rows_by_name)
        except Exception as exc:
            return False, f"Could not rewrite the sheet: {exc}"

        kept = f" {len(rows_by_name)} row(s) kept." if rows_by_name else ""
        return True, f"Columns are now in the app's order.{kept}{note}"

    CURVE_PREFIX = "curve_"

    def save_curve(self, cell_id, epsilon, force_N, fitted_N=None):
        """
        Store one force curve as its own tab in the same spreadsheet.

        A service account owns nothing in Drive: its storage quota is zero
        bytes, so it cannot create a file there however the folder is shared.
        It can write into a spreadsheet somebody else owns, though, and a few
        hundred rows is nothing to a sheet. So the curve lives as a tab beside
        the summary until Box is available for the real archive.

        Returns
        -------
        (success, message, url) : the url points at the tab itself.
        """
        if not self.spreadsheet:
            return False, "Not connected to a spreadsheet.", ""

        safe = "".join(
            ch if (ch.isalnum() or ch in " -_") else "_" for ch in str(cell_id)
        ).strip() or "cell"
        # Google caps sheet titles at 100 characters.
        title = (self.CURVE_PREFIX + safe)[:100]

        rows = [["relative_deformation", "force_N"]
                + (["fit_N"] if fitted_N is not None else [])]
        for index in range(len(epsilon)):
            row = [float(epsilon[index]), float(force_N[index])]
            if fitted_N is not None:
                value = float(fitted_N[index])
                row.append("" if value != value else value)   # NaN outside the fit
            rows.append(row)

        try:
            try:
                worksheet = self.spreadsheet.worksheet(title)
                # Refitting the same cell should replace its curve, not add a
                # second tab with the same name plus a suffix.
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(
                    title=title, rows=max(len(rows) + 10, 100), cols=4
                )
            worksheet.update(values=rows, range_name=f"A1:C{len(rows)}")
        except Exception as exc:
            return False, f"Could not write the curve: {exc}", ""

        url = ""
        try:
            url = f"{self.spreadsheet.url}#gid={worksheet.id}"
        except Exception:
            pass
        return True, f"Curve saved as the tab “{title}” ({len(rows) - 1} points).", url

    def list_curve_tabs(self):
        """Names of the curve tabs already in this spreadsheet."""
        try:
            return [
                ws.title for ws in self.spreadsheet.worksheets()
                if ws.title.startswith(self.CURVE_PREFIX)
            ]
        except Exception:
            return []

    def load_curve(self, cell_id):
        """Read one stored curve back as a DataFrame, or None."""
        safe = "".join(
            ch if (ch.isalnum() or ch in " -_") else "_" for ch in str(cell_id)
        ).strip()
        title = (self.CURVE_PREFIX + safe)[:100]
        try:
            worksheet = self.spreadsheet.worksheet(title)
            values = worksheet.get_all_values()
        except Exception:
            return None
        if len(values) < 2:
            return None
        frame = pd.DataFrame(values[1:], columns=values[0])
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

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
            self.worksheet.append_row(
                [by_name.get(name, "") for name in header],
                value_input_option="USER_ENTERED",
            )

            # The working goes on the second tab, under the same cell id, so
            # the summary tab stays something a person can read across.
            # Failing to write it must not lose the row that did go in, so
            # this is reported rather than raised.
            extra_note = ""
            sheet = self.extra_worksheet()
            if sheet is not None:
                try:
                    extra_header = self._header_row(sheet, self.extra_columns())
                    sheet.append_row(
                        [by_name.get(name, "") for name in extra_header],
                        value_input_option="USER_ENTERED",
                    )
                except Exception as exc:
                    extra_note = f" (the '{self.EXTRA_TAB}' tab was not written: {exc})"
            else:
                extra_note = f" (no '{self.EXTRA_TAB}' tab)"

            return True, (
                f"✅ Cell {cell_data['cell_id']} saved to database" + extra_note
            )

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

            # Convert to DataFrame. The header comes from somebody's
            # spreadsheet, so it can hold blanks and repeats; both are fatal
            # to the table widget and neither is a reason to show nothing.
            headers = self.unique_headers(all_values[0])
            width = len(headers)
            data = [
                (list(row) + [""] * width)[:width] for row in all_values[1:]
            ]

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

    def describe(self) -> str:
        """Which file and which tab this manager is actually writing to."""
        book = getattr(getattr(self, "spreadsheet", None), "title", None)
        tab = getattr(getattr(self, "worksheet", None), "title", None)
        if book and tab:
            return f"“{book}” → tab “{tab}”"
        if tab:
            return f"tab “{tab}”"
        return "connected, but no worksheet was opened"

    def check(self):
        """
        Read the sheet and say what is there, changing nothing.

        A send button that is enabled is not the same as a sheet that can
        be written to: the credentials can be right, the file open, and the
        service account still hold read-only access. This is the cheapest
        way to find that out before a row goes missing.
        """
        if not self.worksheet:
            return False, (
                "Authenticated, but no worksheet is open. Reconnect with the "
                "spreadsheet id in the box above."
            )
        try:
            header = [h for h in self.worksheet.row_values(1) if str(h).strip()]
            rows = max(len(self.worksheet.get_all_values()) - 1, 0)
        except Exception as exc:
            return False, f"Could not read the sheet: {exc}"

        permissions = ""
        try:
            # gspread raises on a write when the share is read-only, and
            # there is no cheap way to ask beforehand, so the header is
            # rewritten with exactly what it already said.
            if header:
                self.worksheet.update(
                    values=[header],
                    range_name=f"A1:{gspread.utils.rowcol_to_a1(1, len(header))}",
                )
                permissions = " Writing works."
        except Exception as exc:
            who = self.service_account_email() or "the service account"
            return False, (
                f"{self.describe()}: readable, but not writable ({exc}). "
                f"Share it with {who} as an **Editor**, not a Viewer."
            )
        return True, (
            f"{self.describe()}: {len(header)} columns, {rows} row(s)."
            + permissions
        )

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
