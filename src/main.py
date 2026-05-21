import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from .react_agent import ReActAgent
from .utils.deepseek_client import DeepSeekClient
from .prompts.system_prompts import SystemPromptManager
from .tools import (
    FileReader,
    DataPreprocessor,
    ColorSchemeManager,
    Plotter,
    ImageSaver,
    ImageOptimizer
)
from .visualizations import (
    create_bar_plot,
    create_scatter_plot,
    create_heatmap,
    create_clustering_heatmap,
    create_volcano_plot
)
from .tools.code_executor import CodeExecutor, CodeGenerator


class TranscriptomeAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        max_iterations: int = 10,
        verbose: bool = True,
        output_dir: str = "."
    ):
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.output_dir = output_dir

        self.data: Optional[pd.DataFrame] = None
        self.current_plot: Optional[Any] = None
        self.last_result: Optional[Dict[str, Any]] = None
        
        self.code_executor = None
        self.code_generator = None
        
        self._setup_tools()
        self._setup_agent()
        self._init_code_tools()

    def _setup_tools(self):
        self.tools = {
            'read_file': self._read_file,
            'get_data_info': self._get_data_info,
            'preprocess_data': self._preprocess_data,
            'filter_genes': self._filter_genes,
            'normalize_data': self._normalize_data,
            'create_bar_plot': self._create_bar_plot,
            'create_scatter_plot': self._create_scatter_plot,
            'create_heatmap': self._create_heatmap,
            'create_clustering_heatmap': self._create_clustering_heatmap,
            'create_volcano_plot': self._create_volcano_plot,
            'save_image': self._save_image,
            'generate_code': self._generate_code,
            'execute_code': self._execute_code,
            'final_answer': self._final_answer,
        }

    def _setup_agent(self):
        prompt_manager = SystemPromptManager()
        system_prompt = prompt_manager.get_system_prompt()

        model_client = DeepSeekClient(
            api_key=self.api_key,
            model=self.model
        )

        self.agent = ReActAgent(
            model_client=model_client,
            tools=self.tools,
            system_prompt=system_prompt,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )

    def _read_file(self, file_path: str, format_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        try:
            self.data = FileReader.read_file(file_path, format_type, **kwargs)
            info = FileReader.get_data_info(self.data)
            validation = FileReader.validate_expression_data(self.data)

            return {
                "success": True,
                "data_shape": info["shape"],
                "columns": info["columns"],
                "numeric_columns": info["numeric_columns"],
                "validation": validation,
                "message": f"Successfully loaded data from {file_path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_data_info(self, **kwargs) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        info = FileReader.get_data_info(self.data)
        validation = FileReader.validate_expression_data(self.data)

        return {
            "success": True,
            "info": info,
            "validation": validation
        }

    def _preprocess_data(
        self,
        operation: str,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        remove_na: bool = True,
        remove_duplicates: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            if operation == 'clean':
                self.data = DataPreprocessor.clean_data(
                    self.data,
                    remove_na=remove_na,
                    remove_duplicates=remove_duplicates
                )
            elif operation == 'normalize':
                if method is None:
                    method = 'log2'
                self.data = DataPreprocessor.normalize(
                    self.data,
                    method=method,
                    columns=columns
                )
            elif operation == 'filter':
                gene_column = self.data.columns[0]
                self.data = DataPreprocessor.filter_genes(
                    self.data,
                    gene_column=gene_column,
                    **kwargs
                )
            elif operation == 'impute':
                self.data = DataPreprocessor.impute_missing_values(
                    self.data,
                    method=method or 'mean'
                )

            return {
                "success": True,
                "operation": operation,
                "new_shape": self.data.shape,
                "message": f"Preprocessing '{operation}' completed"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _filter_genes(
        self,
        gene_column: str,
        min_expression: Optional[float] = None,
        max_expression: Optional[float] = None,
        top_n: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            original_shape = self.data.shape[0]
            self.data = DataPreprocessor.filter_genes(
                self.data,
                gene_column=gene_column,
                min_expression=min_expression,
                max_expression=max_expression,
                top_n=top_n
            )

            return {
                "success": True,
                "original_genes": original_shape,
                "filtered_genes": self.data.shape[0],
                "message": f"Filtered from {original_shape} to {self.data.shape[0]} genes"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _normalize_data(
        self,
        method: str = 'log2',
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            self.data = DataPreprocessor.normalize(
                self.data,
                method=method,
                columns=columns
            )

            return {
                "success": True,
                "method": method,
                "message": f"Normalized using {method} method"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_bar_plot(
        self,
        data: Optional[str] = None,
        x_column: str = "gene",
        y_columns: Optional[List[str]] = None,
        title: str = "Gene Expression Bar Plot",
        xlabel: str = "Genes",
        ylabel: str = "Expression Level",
        palette: str = "Set2",
        gene_filter: Optional[List[str]] = None,
        samples: Optional[List[str]] = None,** kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            plot_data = self.data.copy()
            
            if gene_filter:
                plot_data = plot_data[plot_data[x_column].isin(gene_filter)]
            
            if y_columns is None:
                y_columns = [col for col in plot_data.columns if col != x_column]
            
            if samples:
                y_columns = [col for col in y_columns if col in samples]
                if not y_columns:
                    return {"success": False, "error": "No samples found in data"}
            
            from pathlib import Path
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / f"bar_plot_{timestamp}")
            
            result = create_bar_plot(
                data=plot_data,
                x_column=x_column,
                y_columns=y_columns,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                palette=palette,
                save_path=save_path,** kwargs
            )
            self.current_plot = result['plot_result']['figure']
            self.last_result = result
            return {
                "success": True,
                "message": f"Bar plot created and saved to {save_path}.png",
                "saved_path": f"{save_path}.png",
                "plot_result": result['plot_result']
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_scatter_plot(
        self,
        gene_a: str,
        gene_b: str,
        gene_column: str = "gene",
        title: Optional[str] = None,** kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            gene_a_data = self.data[self.data[gene_column] == gene_a]
            gene_b_data = self.data[self.data[gene_column] == gene_b]
            
            if gene_a_data.empty or gene_b_data.empty:
                return {"success": False, "error": f"Genes {gene_a} or {gene_b} not found in data"}

            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            x_data = gene_a_data[numeric_cols].values.flatten()
            y_data = gene_b_data[numeric_cols].values.flatten()

            if title is None:
                title = f"Correlation: {gene_a} vs {gene_b}"

            from pathlib import Path
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / f"scatter_plot_{gene_a}_{gene_b}_{timestamp}")

            result = create_scatter_plot(
                x_data=x_data,
                y_data=y_data,
                title=title,
                xlabel=f"{gene_a} Expression",
                ylabel=f"{gene_b} Expression",
                save_path=save_path,** kwargs
            )

            self.current_plot = result['plot_result']['figure']
            self.last_result = result
            return {
                "success": True,
                "message": f"Scatter plot created and saved to {save_path}.png",
                "saved_path": f"{save_path}.png",
                "plot_result": result['plot_result']
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_heatmap(
        self,
        data: Optional[str] = None,
        genes: Optional[List[str]] = None,
        samples: Optional[List[str]] = None,
        gene_column: str = "gene",
        title: str = "Gene Expression Heatmap",
        cmap: str = "RdYlBu_r",
        z_score: bool = True,** kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            if genes:
                plot_data = self.data[self.data[gene_column].isin(genes)]
            else:
                plot_data = self.data.copy()
            
            numeric_cols = plot_data.select_dtypes(include=[np.number]).columns
            row_labels = plot_data[gene_column].tolist()
            
            plot_data = plot_data[numeric_cols]
            
            if samples:
                available_samples = [s for s in samples if s in plot_data.columns]
                if available_samples:
                    plot_data = plot_data[available_samples]

            if z_score:
                from scipy import stats
                plot_data = plot_data.apply(stats.zscore)

            from pathlib import Path
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / f"heatmap_{timestamp}")

            result = create_heatmap(
                data=plot_data,
                row_labels=row_labels,
                col_labels=plot_data.columns.tolist() if samples is None else None,
                title=title,
                cmap=cmap,
                save_path=save_path,** kwargs
            )

            self.current_plot = result['plot_result']['figure']
            self.last_result = result
            return {
                "success": True,
                "message": f"Heatmap created and saved to {save_path}.png",
                "saved_path": f"{save_path}.png",
                "plot_result": result['plot_result']
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_clustering_heatmap(
        self,
        data: Optional[str] = None,
        genes: Optional[List[str]] = None,
        samples: Optional[List[str]] = None,
        gene_column: str = "gene",
        title: str = "Hierarchical Clustering Heatmap",
        metric: str = 'euclidean',
        method: str = 'ward',
        cmap: str = "RdYlBu_r",** kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            if genes:
                plot_data = self.data[self.data[gene_column].isin(genes)]
            else:
                plot_data = self.data.copy()
            
            numeric_cols = plot_data.select_dtypes(include=[np.number]).columns
            row_labels = plot_data[gene_column].tolist()
            
            plot_data = plot_data[numeric_cols]
            
            if samples:
                available_samples = [s for s in samples if s in plot_data.columns]
                if available_samples:
                    plot_data = plot_data[available_samples]

            from pathlib import Path
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / f"clustering_heatmap_{timestamp}")

            result = create_clustering_heatmap(
                data=plot_data,
                row_labels=row_labels,
                col_labels=plot_data.columns.tolist() if samples is None else None,
                title=title,
                metric=metric,
                method=method,
                cmap=cmap,
                save_path=save_path,** kwargs
            )

            self.current_plot = result['plot_result']['figure']
            self.last_result = result
            return {
                "success": True,
                "message": f"Clustering heatmap created and saved to {save_path}.png",
                "saved_path": f"{save_path}.png",
                "plot_result": result['plot_result']
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_volcano_plot(
        self,
        log2fc_column: str,
        pvalue_column: str,
        gene_column: str = "gene",
        fc_threshold: float = 1.0,
        pval_threshold: float = 0.05,
        title: str = "Volcano Plot",** kwargs
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded"}

        try:
            if gene_column not in self.data.columns:
                gene_column = self.data.columns[0]
            if log2fc_column not in self.data.columns or pvalue_column not in self.data.columns:
                return {"success": False, "error": "Specified columns not found in data"}

            log2fc = self.data[log2fc_column].values
            pvalue = self.data[pvalue_column].values
            gene_labels = self.data[gene_column].tolist()

            from pathlib import Path
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / f"volcano_plot_{timestamp}")

            result = create_volcano_plot(
                log2fc=log2fc,
                pvalue=pvalue,
                gene_labels=gene_labels,
                fc_threshold=fc_threshold,
                pval_threshold=pval_threshold,
                title=title,
                save_path=save_path,** kwargs
            )

            self.current_plot = result['plot_result']['figure']
            self.last_result = result
            return {
                "success": True,
                "message": f"Volcano plot created and saved to {save_path}.png",
                "saved_path": f"{save_path}.png",
                "plot_result": result['plot_result']
            }
        except Exception as e:
            return {"success": False, "error": "Volcano plot error: " + str(e)}

    def _save_image(
        self,
        file_path: str,
        formats: Optional[List[str]] = None,
        dpi: int = 300,** kwargs
    ) -> Dict[str, Any]:
        if self.current_plot is None:
            return {"success": False, "error": "No plot to save"}

        try:
            from pathlib import Path
            from .tools.image_saver import ImageSaver
            
            full_path = Path(self.output_dir) / Path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            saved_paths = ImageSaver.save_figure(
                self.current_plot,
                str(full_path),
                formats=formats,
                dpi=dpi
            )

            return {
                "success": True,
                "saved_paths": saved_paths,
                "message": f"Image saved to: {', '.join(saved_paths)}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _final_answer(self, answer: str, **kwargs) -> Dict[str, Any]:
        return {
            "success": True,
            "answer": answer,
            "final": True
        }
    
    def _init_code_tools(self):
        """初始化代码生成和执行工具"""
        if self.code_executor is None:
            self.code_executor = CodeExecutor(
                data=self.data,
                output_dir=self.output_dir
            )
        
        if self.code_generator is None:
            model_client = DeepSeekClient(
                api_key=self.api_key,
                model=self.model
            )
            self.code_generator = CodeGenerator(model_client)
    
    def _generate_code(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """根据任务描述生成Python代码"""
        try:
            self._init_code_tools()
            
            data_info = None
            if self.data is not None:
                data_info = {
                    'shape': self.data.shape,
                    'columns': list(self.data.columns),
                    'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()}
                }
            
            code = self.code_generator.generate(
                task_description=task_description,
                data_info=data_info
            )
            
            if not code:
                return {
                    "success": False,
                    "error": "Failed to generate code"
                }
            
            self.last_result = {
                "generated_code": code,
                "task_description": task_description
            }
            
            return {
                "success": True,
                "generated_code": code,
                "message": "Code generated successfully. Now use execute_code to run it."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Code generation error: {str(e)}"
            }
    
    def _execute_code(self, code: str, **kwargs) -> Dict[str, Any]:
        """执行生成的Python代码"""
        try:
            self._init_code_tools()
            
            self.code_executor.data = self.data
            
            result = self.code_executor.execute(code)
            
            if result.get('success') and result.get('data') is not None:
                self.data = result['data']
                self.code_executor.data = self.data
            
            self.last_result = result
            
            if result.get('success'):
                output_msg = f"Code executed successfully"
                if result.get('output'):
                    output_msg += f"\nOutput:\n{result['output'][:500]}"
                
                return {
                    "success": True,
                    "result": result.get('result'),
                    "output": result.get('output'),
                    "message": output_msg
                }
            else:
                return {
                    "success": False,
                    "error": result.get('error'),
                    "message": f"Code execution failed: {result.get('error')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Code execution error: {str(e)}"
            }

    def run(self, user_request: str) -> Dict[str, Any]:
        result = self.agent.run(user_request)
        return result

    def load_data(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return self._read_file(file_path, **kwargs)


def run_agent(
    api_key: str,
    user_request: str,
    data_path: Optional[str] = None,
    model: str = "deepseek-chat",
    verbose: bool = True,
    output_dir: str = "."
) -> Dict[str, Any]:
    agent = TranscriptomeAgent(
        api_key=api_key,
        model=model,
        verbose=verbose,
        output_dir=output_dir
    )

    if data_path:
        load_result = agent.load_data(data_path)
        if not load_result.get('success'):
            return {
                "success": False,
                "error": f"Failed to load data: {load_result.get('error')}"
            }
        
        from src.react_agent.core import Thought, Action, Observation, ReActStep
        
        thought = Thought(content=f"Data has been loaded from {data_path}. Shape: {load_result.get('data_shape')}. Columns: {load_result.get('columns')[:5]}...")
        action = Action(tool_name='read_file', tool_input={'file_path': data_path})
        observation = Observation(
            tool_name='read_file',
            result=load_result,
            success=True
        )
        step = ReActStep(thought=thought, action=action, observation=observation)
        agent.agent.history.append(step)

    result = agent.run(user_request)
    return result
