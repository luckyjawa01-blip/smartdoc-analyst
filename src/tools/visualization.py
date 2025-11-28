"""Visualization Tool for SmartDoc Analyst.

This tool generates data visualizations including charts,
graphs, and diagrams for document analysis outputs.
"""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult


class VisualizationTool(BaseTool):
    """Data visualization generation tool.
    
    Creates various types of visualizations including bar charts,
    line graphs, pie charts, and word clouds from analysis data.
    
    Attributes:
        default_format: Default output format (png, svg, base64).
        default_style: Default visual style.
        
    Example:
        >>> tool = VisualizationTool()
        >>> result = await tool.execute(
        ...     data={"A": 10, "B": 20, "C": 30},
        ...     chart_type="bar",
        ...     title="Sample Distribution"
        ... )
    """
    
    def __init__(
        self,
        default_format: str = "base64",
        default_style: str = "default"
    ):
        """Initialize the visualization tool.
        
        Args:
            default_format: Default output format.
            default_style: Default visual style.
        """
        super().__init__(
            name="visualization",
            description="Generate data visualizations and charts"
        )
        self.default_format = default_format
        self.default_style = default_style
        self._matplotlib_available = None
        
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available.
        
        Returns:
            bool: True if matplotlib is importable.
        """
        if self._matplotlib_available is None:
            try:
                import matplotlib
                matplotlib.use('Agg')
                self._matplotlib_available = True
            except ImportError:
                self._matplotlib_available = False
        return self._matplotlib_available
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Generate a visualization.
        
        Args:
            data: Data to visualize (dict, list, or DataFrame).
            chart_type: Type of chart (bar, line, pie, scatter, heatmap).
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            output_format: Output format (base64, bytes, path).
            
        Returns:
            ToolResult: Visualization result.
        """
        data = kwargs.get("data", {})
        chart_type = kwargs.get("chart_type", "bar")
        title = kwargs.get("title", "")
        xlabel = kwargs.get("xlabel", "")
        ylabel = kwargs.get("ylabel", "")
        output_format = kwargs.get("output_format", self.default_format)
        
        if not data:
            return ToolResult(
                success=False,
                error="Data is required for visualization"
            )
            
        if not self._check_matplotlib():
            return self._text_visualization(data, chart_type, title)
            
        try:
            if chart_type == "bar":
                result = self._create_bar_chart(data, title, xlabel, ylabel)
            elif chart_type == "line":
                result = self._create_line_chart(data, title, xlabel, ylabel)
            elif chart_type == "pie":
                result = self._create_pie_chart(data, title)
            elif chart_type == "scatter":
                result = self._create_scatter_plot(data, title, xlabel, ylabel)
            elif chart_type == "heatmap":
                result = self._create_heatmap(data, title)
            else:
                result = self._create_bar_chart(data, title, xlabel, ylabel)
                
            return ToolResult(
                success=True,
                data={
                    "image": result,
                    "format": output_format,
                    "chart_type": chart_type,
                    "title": title
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Visualization failed: {str(e)}"
            )
            
    def _create_bar_chart(
        self,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str
    ) -> str:
        """Create a bar chart.
        
        Args:
            data: Data dictionary with labels and values.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            
        Returns:
            str: Base64 encoded image.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(data, dict):
            labels = list(data.keys())
            values = list(data.values())
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                labels = [d.get("label", str(i)) for i, d in enumerate(data)]
                values = [d.get("value", 0) for d in data]
            else:
                labels = [str(i) for i in range(len(data))]
                values = data
        else:
            labels = ["Data"]
            values = [data]
            
        # Create bars with color gradient
        colors = plt.cm.viridis([i/len(labels) for i in range(len(labels))])
        bars = ax.bar(labels, values, color=colors)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}' if isinstance(value, float) else str(value),
                    ha='center', va='bottom', fontsize=9)
            
        ax.set_xlabel(xlabel or "Categories")
        ax.set_ylabel(ylabel or "Values")
        ax.set_title(title or "Bar Chart")
        
        # Rotate labels if many
        if len(labels) > 5:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
        
    def _create_line_chart(
        self,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str
    ) -> str:
        """Create a line chart.
        
        Args:
            data: Data for the line chart.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            
        Returns:
            str: Base64 encoded image.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(data, dict):
            x = list(data.keys())
            y = list(data.values())
        elif isinstance(data, list):
            x = list(range(len(data)))
            y = data
        else:
            x = [0]
            y = [data]
            
        ax.plot(x, y, marker='o', linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3)
        
        ax.set_xlabel(xlabel or "X")
        ax.set_ylabel(ylabel or "Y")
        ax.set_title(title or "Line Chart")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
        
    def _create_pie_chart(
        self,
        data: Dict[str, Any],
        title: str
    ) -> str:
        """Create a pie chart.
        
        Args:
            data: Data dictionary with labels and values.
            title: Chart title.
            
        Returns:
            str: Base64 encoded image.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if isinstance(data, dict):
            labels = list(data.keys())
            values = list(data.values())
        else:
            labels = [f"Item {i}" for i in range(len(data))]
            values = data
            
        # Filter out zero/negative values
        filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
        if filtered:
            labels, values = zip(*filtered)
            
        colors = plt.cm.Set3([i/len(labels) for i in range(len(labels))])
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(values)
        )
        
        ax.set_title(title or "Pie Chart")
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
        
    def _create_scatter_plot(
        self,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str
    ) -> str:
        """Create a scatter plot.
        
        Args:
            data: Data with x and y coordinates.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            
        Returns:
            str: Base64 encoded image.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Parse data
        if isinstance(data, dict):
            if "x" in data and "y" in data:
                x = data["x"]
                y = data["y"]
            else:
                x = list(range(len(data)))
                y = list(data.values())
        elif isinstance(data, list):
            if data and isinstance(data[0], (list, tuple)):
                x = [p[0] for p in data]
                y = [p[1] for p in data]
            else:
                x = list(range(len(data)))
                y = data
        else:
            x = [0]
            y = [data]
            
        ax.scatter(x, y, alpha=0.6, s=50)
        
        ax.set_xlabel(xlabel or "X")
        ax.set_ylabel(ylabel or "Y")
        ax.set_title(title or "Scatter Plot")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
        
    def _create_heatmap(
        self,
        data: Dict[str, Any],
        title: str
    ) -> str:
        """Create a heatmap.
        
        Args:
            data: 2D data for heatmap.
            title: Chart title.
            
        Returns:
            str: Base64 encoded image.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert data to 2D array
        if isinstance(data, dict):
            matrix = list(data.values())
            if all(isinstance(v, (list, tuple)) for v in matrix):
                matrix = np.array(matrix)
            else:
                matrix = np.array([list(data.values())])
        elif isinstance(data, list):
            if data and isinstance(data[0], (list, tuple)):
                matrix = np.array(data)
            else:
                matrix = np.array([data])
        else:
            matrix = np.array([[data]])
            
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax)
        
        ax.set_title(title or "Heatmap")
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
        
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string.
        
        Args:
            fig: Matplotlib figure.
            
        Returns:
            str: Base64 encoded PNG.
        """
        import matplotlib.pyplot as plt
        
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close(fig)
        
        return image_base64
        
    def _text_visualization(
        self,
        data: Dict[str, Any],
        chart_type: str,
        title: str
    ) -> ToolResult:
        """Generate text-based visualization when matplotlib unavailable.
        
        Args:
            data: Data to visualize.
            chart_type: Type of chart.
            title: Chart title.
            
        Returns:
            ToolResult: Text visualization.
        """
        lines = [f"=== {title or 'Data Visualization'} ===", ""]
        
        if isinstance(data, dict):
            max_label_len = max(len(str(k)) for k in data.keys()) if data else 0
            max_value = max(data.values()) if data else 1
            
            for label, value in data.items():
                bar_len = int((value / max_value) * 40) if max_value > 0 else 0
                bar = "█" * bar_len
                lines.append(f"{str(label):<{max_label_len}} | {bar} {value}")
        else:
            lines.append(str(data))
            
        return ToolResult(
            success=True,
            data={
                "text_visualization": "\n".join(lines),
                "format": "text",
                "chart_type": chart_type,
                "title": title,
                "note": "matplotlib not available, showing text visualization"
            }
        )
        
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: Parameter schema.
        """
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": ["object", "array"],
                    "description": "Data to visualize"
                },
                "chart_type": {
                    "type": "string",
                    "description": "Type of visualization",
                    "enum": ["bar", "line", "pie", "scatter", "heatmap"],
                    "default": "bar"
                },
                "title": {
                    "type": "string",
                    "description": "Chart title"
                },
                "xlabel": {
                    "type": "string",
                    "description": "X-axis label"
                },
                "ylabel": {
                    "type": "string",
                    "description": "Y-axis label"
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format",
                    "enum": ["base64", "bytes", "path"],
                    "default": "base64"
                }
            },
            "required": ["data"]
        }
