import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data import parliament_results, local_results, poll_data


warnings.filterwarnings("ignore")


class NorwegianPoliticalAnalyzer:
    def __init__(self):
        """Initialize the Political Analysis Framework"""
        self.parliament_data = None
        self.local_data = None
        self.poll_data = None
        self.combined_data = None

    def load_data(self, parliament_results, local_results, poll_data=None):
        """Load and structure the political data"""

        # Parliament election data
        self.parliament_data = pd.DataFrame(
            [
                {"year": year, "result": result, "type": "parliament"}
                for year, result in parliament_results.items()
            ]
        )

        # Local election data
        local_rows = []
        for year, results in local_results.items():
            local_rows.append(
                {"year": year, "result": results["municipal"], "type": "municipal"}
            )
            local_rows.append(
                {"year": year, "result": results["county"], "type": "county"}
            )
        self.local_data = pd.DataFrame(local_rows)

        # Poll data (if provided) - Updated for FrP historical data
        if poll_data:
            self.poll_data = pd.DataFrame(poll_data)
            # Parse the date format for FrP data (e.g., "Jan 10" -> year based on chronological order)
            # This regex needs to handle cases like "Jan 10" becoming "Jan 2010"
            self.poll_data["parsed_date"] = self.poll_data["date"].apply(
                lambda x: datetime.strptime(
                    f"{x.split(' ')[0]} 20{x.split(' ')[1]}", "%b %Y"
                )
            )

            # Create a proper timeline based on chronological order
            self.poll_data = self.poll_data.sort_values("parsed_date").reset_index(
                drop=True
            )
            self.poll_data["timeline_order"] = range(len(self.poll_data))

        # Combine all election data
        self.combined_data = pd.concat(
            [self.parliament_data, self.local_data], ignore_index=True
        )

        print("✅ FrP Political Data loaded successfully!")
        print(
            f"Parliament elections: {len(self.parliament_data)} records ({self.parliament_data['year'].min()}-{self.parliament_data['year'].max()})"
        )
        print(
            f"Local elections: {len(self.local_data)} records ({self.local_data['year'].min()}-{self.local_data['year'].max()})"
        )
        if poll_data is not None:
            print(
                f"Poll data: {len(self.poll_data)} records ({self.poll_data['date'].iloc[0]} to {self.poll_data['date'].iloc[-1]})"
            )
            print(
                f"Poll range: {self.poll_data['FrP'].min():.1f}% - {self.poll_data['FrP'].max():.1f}%"
            )

    def basic_statistics(self):
        """Generate basic descriptive statistics"""
        print("\n" + "=" * 60)
        print("BASIC STATISTICS")
        print("=" * 60)

        stats_summary = (
            self.combined_data.groupby("type")["result"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(2)
        )

        print("\nDescriptive Statistics by Election Type:")
        print(stats_summary)

        # Calculate volatility (standard deviation of changes)
        for election_type in ["parliament", "municipal", "county"]:
            subset = self.combined_data[
                self.combined_data["type"] == election_type
            ].sort_values("year")
            if len(subset) > 1:
                changes = subset["result"].diff().dropna()
                volatility = changes.std()
                print(f"\nVolatility ({election_type}): {volatility:.2f}")

        return stats_summary

    def trend_analysis(self):
        """Analyze long-term trends"""
        print("\n" + "=" * 60)
        print("TREND ANALYSIS")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Norwegian Political Party - Trend Analysis", fontsize=16, fontweight="bold"
        )

        # 1. Overall timeline
        ax1 = axes[0, 0]
        for election_type in ["parliament", "municipal", "county"]:
            subset = self.combined_data[self.combined_data["type"] == election_type]
            ax1.plot(
                subset["year"],
                subset["result"],
                marker="o",
                label=election_type,
                linewidth=2,
            )
        ax1.set_title("Electoral Performance Over Time")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Vote Share (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Parliament elections with trend line
        ax2 = axes[0, 1]
        parl_data = self.parliament_data.sort_values("year")
        ax2.scatter(parl_data["year"], parl_data["result"], alpha=0.7, s=80)

        # Add trend line
        z = np.polyfit(parl_data["year"], parl_data["result"], 1)
        p = np.poly1d(z)
        ax2.plot(parl_data["year"], p(parl_data["year"]), "r--", alpha=0.8)
        ax2.set_title(f"Parliament Elections (Trend: {z[0]:.2f}% per year)")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Vote Share (%)")
        ax2.grid(True, alpha=0.3)

        # 3. Local vs National comparison
        ax3 = axes[1, 0]
        # Get overlapping years
        parl_years = set(self.parliament_data["year"])
        local_municipal = self.local_data[self.local_data["type"] == "municipal"]
        local_county = self.local_data[self.local_data["type"] == "county"]

        ax3.scatter(
            local_municipal["year"],
            local_municipal["result"],
            alpha=0.7,
            label="Municipal",
            s=60,
        )
        ax3.scatter(
            local_county["year"],
            local_county["result"],
            alpha=0.7,
            label="County",
            s=60,
        )
        ax3.set_title("Local Elections Performance")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Vote Share (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Performance distribution
        ax4 = axes[1, 1]
        self.combined_data.boxplot(column="result", by="type", ax=ax4)
        ax4.set_title("Performance Distribution by Election Type")
        ax4.set_xlabel("Election Type")
        ax4.set_ylabel("Vote Share (%)")

        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """Analyze correlations between different election types"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

        # Create pivot table for correlation analysis
        pivot_data = self.combined_data.pivot(
            index="year", columns="type", values="result"
        )

        # Fill missing values with interpolation where possible
        pivot_data = pivot_data.interpolate()

        # Calculate correlation matrix
        correlation_matrix = pivot_data.corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))

        # Visualize correlations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=axes[0],
        )
        axes[0].set_title("Correlation Heatmap")

        # Scatter plots for strongest correlations
        if "parliament" in pivot_data.columns and "municipal" in pivot_data.columns:
            # Remove NaN values for scatter plot
            clean_data = pivot_data[["parliament", "municipal"]].dropna()
            if len(clean_data) > 0:
                axes[1].scatter(
                    clean_data["parliament"], clean_data["municipal"], alpha=0.7, s=80
                )

                # Add trend line
                if len(clean_data) > 1:
                    z = np.polyfit(clean_data["parliament"], clean_data["municipal"], 1)
                    p = np.poly1d(z)
                    axes[1].plot(
                        clean_data["parliament"],
                        p(clean_data["parliament"]),
                        "r--",
                        alpha=0.8,
                    )

                axes[1].set_xlabel("Parliament Results (%)")
                axes[1].set_ylabel("Municipal Results (%)")
                axes[1].set_title("Parliament vs Municipal Performance")
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return correlation_matrix

    def electoral_phases(self):
        """Identify distinct electoral phases"""
        print("\n" + "=" * 60)
        print("ELECTORAL PHASES ANALYSIS")
        print("=" * 60)

        parl_data = self.parliament_data.sort_values("year")

        # Define phases based on performance levels
        phases = []
        for _, row in parl_data.iterrows():
            if row["result"] < 5:
                phase = "Marginal (< 5%)"
            elif row["result"] < 15:
                phase = "Moderate (5-15%)"
            else:
                phase = "Strong (> 15%)"
            phases.append(phase)

        parl_data["phase"] = phases

        print("\nElectoral Phases:")
        phase_summary = (
            parl_data.groupby("phase")
            .agg({"year": ["min", "max", "count"], "result": ["mean", "std"]})
            .round(2)
        )
        print(phase_summary)

        # Visualize phases
        plt.figure(figsize=(12, 6))
        colors = {
            "Marginal (< 5%)": "red",
            "Moderate (5-15%)": "orange",
            "Strong (> 15%)": "green",
        }

        for phase in parl_data["phase"].unique():
            subset = parl_data[parl_data["phase"] == phase]
            plt.scatter(
                subset["year"],
                subset["result"],
                c=colors[phase],
                label=phase,
                s=100,
                alpha=0.7,
            )

        plt.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5% threshold")
        plt.axhline(
            y=15, color="green", linestyle="--", alpha=0.5, label="15% threshold"
        )
        plt.xlabel("Year")
        plt.ylabel("Vote Share (%)")
        plt.title("Electoral Phases Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return parl_data

    # ...existing code...

    def poll_trend_analysis(self):
        """Analyze and visualize polling trends for FrP"""
        if self.poll_data is None:
            print("No poll data loaded.")
            return

        print("\n" + "=" * 60)
        print("POLL TREND ANALYSIS")
        print("=" * 60)

        plt.figure(figsize=(14, 6))
        plt.plot(
            self.poll_data["parsed_date"],
            self.poll_data["FrP"],
            marker="o",
            linestyle="-",
            color="blue",
            label="FrP Poll (%)",
        )

        # Add rolling average for smoothing
        if len(self.poll_data) >= 3:
            self.poll_data["FrP_rolling"] = (
                self.poll_data["FrP"].rolling(window=3, min_periods=1).mean()
            )
            plt.plot(
                self.poll_data["parsed_date"],
                self.poll_data["FrP_rolling"],
                color="orange",
                linestyle="--",
                label="3-period Rolling Avg",
            )

        plt.title("FrP Polling Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Poll (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def interactive_election_plot(self):
        """Create an interactive Plotly plot for election results"""
        if self.combined_data is None:
            print("No election data loaded.")
            return

        print("\n" + "=" * 60)
        print("INTERACTIVE ELECTION PLOT")
        print("=" * 60)

        fig = px.line(
            self.combined_data,
            x="year",
            y="result",
            color="type",
            markers=True,
            title="Norwegian Political Party - Election Results Over Time",
            labels={
                "result": "Vote Share (%)",
                "year": "Year",
                "type": "Election Type",
            },
        )
        fig.update_layout(legend_title_text="Election Type")
        fig.show()

    def run_full_analysis(self):
        """Run all analyses in sequence"""
        self.basic_statistics()
        self.trend_analysis()
        self.correlation_analysis()
        self.electoral_phases()
        if self.poll_data is not None:
            self.poll_trend_analysis()
        self.interactive_election_plot()


# ...existing code...

if __name__ == "__main__":
    # Example data (replace with your actual data)
    parliament_results = parliament_results

    local_results = local_results
    poll_data = poll_data
    analyzer = NorwegianPoliticalAnalyzer()
    analyzer.load_data(parliament_results, local_results, poll_data)
    analyzer.run_full_analysis()
