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
            self.poll_data["parsed_date"] = pd.to_datetime(
                self.poll_data["date"].str.replace(
                    r"(\w{3}) (\d{2})", r"\1 20\2", regex=True
                ),
                format="%b %Y",
                errors="coerce",
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
        if poll_data:
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

    def volatility_analysis(self):
        """Analyze electoral volatility"""
        print("\n" + "=" * 60)
        print("VOLATILITY ANALYSIS")
        print("=" * 60)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, election_type in enumerate(["parliament", "municipal", "county"]):
            subset = self.combined_data[
                self.combined_data["type"] == election_type
            ].sort_values("year")

            if len(subset) > 1:
                # Calculate election-to-election changes
                changes = subset["result"].diff().dropna()
                years = subset["year"].iloc[1:]  # Skip first year as it has no change

                axes[i].bar(
                    years,
                    changes,
                    alpha=0.7,
                    color=["green" if x > 0 else "red" for x in changes],
                )
                axes[i].axhline(y=0, color="black", linestyle="-", alpha=0.3)
                axes[i].set_title(f"{election_type.capitalize()} - Election Changes")
                axes[i].set_xlabel("Year")
                axes[i].set_ylabel("Change in Vote Share (%)")
                axes[i].grid(True, alpha=0.3)

                # Add statistics
                volatility = changes.std()
                mean_change = changes.mean()
                axes[i].text(
                    0.05,
                    0.95,
                    f"Volatility: {volatility:.2f}%\nMean Change: {mean_change:.2f}%",
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()
        plt.show()

    def predictive_modeling(self):
        """Build predictive models for future performance"""
        print("\n" + "=" * 60)
        print("PREDICTIVE MODELING")
        print("=" * 60)

        parl_data = self.parliament_data.sort_values("year")
        X = parl_data["year"].values.reshape(-1, 1)
        y = parl_data["result"].values

        # Linear regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)

        # Polynomial regression (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)

        # Predictions
        future_years = np.arange(2025, 2034).reshape(-1, 1)
        linear_pred = linear_model.predict(future_years)
        poly_pred = poly_model.predict(poly_features.transform(future_years))

        # Model evaluation
        linear_r2 = r2_score(y, linear_model.predict(X))
        poly_r2 = r2_score(y, poly_model.predict(X_poly))

        print(f"Linear Model R²: {linear_r2:.3f}")
        print(f"Polynomial Model R²: {poly_r2:.3f}")

        # Visualization
        plt.figure(figsize=(12, 8))

        # Historical data
        plt.scatter(
            parl_data["year"],
            parl_data["result"],
            color="blue",
            s=100,
            alpha=0.7,
            label="Historical Results",
        )

        # Model fits
        year_range = np.arange(1973, 2034)
        linear_fit = linear_model.predict(year_range.reshape(-1, 1))
        poly_fit = poly_model.predict(
            poly_features.transform(year_range.reshape(-1, 1))
        )

        plt.plot(
            year_range,
            linear_fit,
            "r--",
            alpha=0.8,
            label=f"Linear Trend (R²={linear_r2:.3f})",
        )
        plt.plot(
            year_range,
            poly_fit,
            "g--",
            alpha=0.8,
            label=f"Polynomial Trend (R²={poly_r2:.3f})",
        )

        # Future predictions
        plt.scatter(
            future_years.flatten(),
            linear_pred,
            color="red",
            s=80,
            alpha=0.7,
            marker="^",
            label="Linear Predictions",
        )
        plt.scatter(
            future_years.flatten(),
            poly_pred,
            color="green",
            s=80,
            alpha=0.7,
            marker="s",
            label="Polynomial Predictions",
        )

        plt.axvline(
            x=2025, color="black", linestyle=":", alpha=0.5, label="Prediction Start"
        )
        plt.xlabel("Year")
        plt.ylabel("Vote Share (%)")
        plt.title("Electoral Performance - Historical Data and Future Predictions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Print predictions
        print("\nFuture Predictions:")
        pred_df = pd.DataFrame(
            {
                "Year": future_years.flatten(),
                "Linear_Prediction": linear_pred,
                "Polynomial_Prediction": poly_pred,
            }
        )
        print(pred_df.round(2))

        return pred_df

    def poll_analysis(self):
        """Comprehensive analysis of FrP polling data"""
        if self.poll_data is None:
            print("No poll data available for analysis")
            return

        print("\n" + "=" * 60)
        print("FrP POLLING ANALYSIS (2010-2025)")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            "FrP Polling Analysis - Comprehensive Overview",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Full timeline with key events
        ax1 = axes[0, 0]
        ax1.plot(
            self.poll_data["timeline_order"],
            self.poll_data["FrP"],
            linewidth=2,
            alpha=0.8,
            color="blue",
        )
        ax1.fill_between(
            self.poll_data["timeline_order"],
            self.poll_data["FrP"],
            alpha=0.3,
            color="lightblue",
        )

        # Mark election years on poll timeline
        election_years = [2013, 2017, 2021, 2025]
        for year in election_years:
            if year <= 2025:  # Only show up to current data
                year_polls = self.poll_data[
                    self.poll_data["parsed_date"].dt.year == year
                ]
                if not year_polls.empty:
                    ax1.axvline(
                        x=year_polls["timeline_order"].mean(),
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                    )
                    ax1.text(
                        year_polls["timeline_order"].mean(),
                        ax1.get_ylim()[1] * 0.9,
                        f"{year} Election",
                        rotation=90,
                        ha="right",
                    )

        ax1.set_title("FrP Poll Evolution (2010-2025)")
        ax1.set_xlabel("Timeline")
        ax1.set_ylabel("Poll Support (%)")
        ax1.grid(True, alpha=0.3)

        # 2. Yearly averages and volatility
        ax2 = axes[0, 1]
        yearly_stats = (
            self.poll_data.groupby(self.poll_data["parsed_date"].dt.year)["FrP"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        yearly_stats = yearly_stats.dropna()

        ax2.errorbar(
            yearly_stats["parsed_date"],
            yearly_stats["mean"],
            yerr=yearly_stats["std"],
            fmt="o-",
            capsize=5,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )
        ax2.fill_between(
            yearly_stats["parsed_date"],
            yearly_stats["min"],
            yearly_stats["max"],
            alpha=0.2,
            color="orange",
        )
        ax2.set_title("Yearly Poll Averages with Volatility")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Poll Support (%)")
        ax2.grid(True, alpha=0.3)

        # 3. Seasonal patterns
        ax3 = axes[1, 0]
        self.poll_data["month"] = self.poll_data["parsed_date"].dt.month
        monthly_avg = self.poll_data.groupby("month")["FrP"].mean()
        monthly_std = self.poll_data.groupby("month")["FrP"].std()

        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        ax3.bar(range(1, 13), monthly_avg, alpha=0.7, color="green")
        ax3.errorbar(
            range(1, 13),
            monthly_avg,
            yerr=monthly_std,
            fmt="none",
            capsize=3,
            color="black",
            alpha=0.7,
        )
        ax3.set_title("Seasonal Polling Patterns")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Average Poll Support (%)")
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months, rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Distribution and key statistics
        ax4 = axes[1, 1]
        ax4.hist(
            self.poll_data["FrP"], bins=20, alpha=0.7, color="purple", edgecolor="black"
        )
        ax4.axvline(
            self.poll_data["FrP"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.poll_data['FrP'].mean():.1f}%",
        )
        ax4.axvline(
            self.poll_data["FrP"].median(),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {self.poll_data['FrP'].median():.1f}%",
        )
        ax4.set_title("FrP Poll Support Distribution")
        ax4.set_xlabel("Poll Support (%)")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Key insights
        print("\n📊 KEY POLLING INSIGHTS:")
        print(f"Average poll support: {self.poll_data['FrP'].mean():.1f}%")
        print(
            f"Peak support: {self.poll_data['FrP'].max():.1f}% ({self.poll_data.loc[self.poll_data['FrP'].idxmax(), 'date']})"
        )
        print(
            f"Lowest support: {self.poll_data['FrP'].min():.1f}% ({self.poll_data.loc[self.poll_data['FrP'].idxmin(), 'date']})"
        )
        print(f"Volatility (std dev): {self.poll_data['FrP'].std():.1f}%")
        print(
            f"Support range: {self.poll_data['FrP'].max() - self.poll_data['FrP'].min():.1f} percentage points"
        )

        # Recent trend analysis
        recent_data = self.poll_data.tail(12)  # Last 12 polls
        recent_trend = recent_data["FrP"].iloc[-1] - recent_data["FrP"].iloc[0]
        print(f"\nRecent trend (last 12 polls): {recent_trend:+.1f} percentage points")

        return yearly_stats

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE POLITICAL ANALYSIS REPORT")
        print("=" * 80)

        # Run all analyses
        basic_stats = self.basic_statistics()
        self.trend_analysis()
        correlations = self.correlation_analysis()
        phases = self.electoral_phases()
        self.volatility_analysis()
        predictions = self.predictive_modeling()
        self.poll_analysis()

        print("\n" + "=" * 80)
        print("REPORT COMPLETE")
        print("=" * 80)


# Example usage and data setup
def main():
    # Your data
    parliament_election_results = {
        1973: 5.0,
        1977: 1.9,
        1981: 4.5,
        1985: 3.7,
        1989: 13.0,
        1993: 6.3,
        1997: 15.3,
        2001: 14.6,
        2005: 22.1,
        2009: 22.9,
        2013: 16.3,
        2017: 15.3,
        2021: 11.7,
        2025: 23.8,
    }

    local_election_results = {
        1975: {"municipal": 0.8, "county": 1.4},
        1979: {"municipal": 1.9, "county": 2.5},
        1983: {"municipal": 5.3, "county": 6.3},
        1987: {"municipal": 10.4, "county": 12.3},
        1991: {"municipal": 6.5, "county": 7.0},
        1995: {"municipal": 10.5, "county": 12.0},
        1999: {"municipal": 12.1, "county": 13.4},
        2003: {"municipal": 16.4, "county": 17.9},
        2007: {"municipal": 17.5, "county": 18.5},
        2011: {"municipal": 11.4, "county": 11.8},
        2015: {"municipal": 9.5, "county": 10.2},
        2019: {"municipal": 8.2, "county": 8.7},
        2023: {"municipal": 11.4, "county": 12.5},
    }

    # Initialize analyzer
    analyzer = NorwegianPoliticalAnalyzer()

    # Load your data (add poll_data if you want to include the historical poll data)
    analyzer.load_data(parliament_election_results, local_election_results)

    # Run comprehensive analysis
    analyzer.generate_report()

    print("\n🎯 Analysis Framework Ready!")
    print("You can now run individual analyses:")
    print("- analyzer.basic_statistics()")
    print("- analyzer.trend_analysis()")
    print("- analyzer.correlation_analysis()")
    print("- analyzer.electoral_phases()")
    print("- analyzer.volatility_analysis()")
    print("- analyzer.predictive_modeling()")


if __name__ == "__main__":
    main()
