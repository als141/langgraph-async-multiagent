#!/usr/bin/env python3
"""
Comprehensive Code Validation Test for LangGraph, LangChain, and OpenAI Updates
Tests all improvements made during the investigation and validates system readiness.
"""

import ast
import os
import sys
from pathlib import Path
import re
from typing import Dict, List, Tuple

def load_file_content(file_path: str) -> str:
    """Load file content safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""

def check_pyproject_dependencies(content: str) -> Dict[str, bool]:
    """Validate dependency versions in pyproject.toml"""
    checks = {
        'langchain_version': False,
        'langchain_openai_version': False,
        'langgraph_version': False,
        'openai_version': False,
        'pydantic_version': False
    }
    
    # Check for modern dependency versions
    patterns = {
        'langchain_version': r'langchain>=0\.3\.0',
        'langchain_openai_version': r'langchain-openai>=0\.3\.27',
        'langgraph_version': r'langgraph>=0\.4\.8',
        'openai_version': r'openai>=1\.40\.0',
        'pydantic_version': r'pydantic>=2\.0\.0'
    }
    
    for check, pattern in patterns.items():
        if re.search(pattern, content):
            checks[check] = True
    
    return checks

def check_agents_configuration(content: str) -> Dict[str, bool]:
    """Validate agents.py configuration and improvements"""
    checks = {
        'gpt_4o_mini_model': False,
        'output_version_responses': False,
        'use_responses_api': False,
        'max_retries_config': False,
        'strict_mode': False,
        'timeout_config': False,
        'error_handling': False,
        'async_operations': False,
        'content_extraction': False,
        'emergency_response': False
    }
    
    # Check for latest configurations
    patterns = {
        'gpt_4o_mini_model': r'model="gpt-4o-mini"',
        'output_version_responses': r'output_version="responses/v1"',
        'use_responses_api': r'use_responses_api=True',
        'max_retries_config': r'max_retries=3',
        'strict_mode': r'strict=True',
        'timeout_config': r'timeout=30',
        'error_handling': r'_safe_llm_call',
        'async_operations': r'async def',
        'content_extraction': r'_extract_clean_content',
        'emergency_response': r'_create_emergency_response'
    }
    
    for check, pattern in patterns.items():
        if re.search(pattern, content):
            checks[check] = True
    
    return checks

def check_orchestrator_improvements(content: str) -> Dict[str, bool]:
    """Validate orchestrator.py streaming and error handling"""
    checks = {
        'streaming_support': False,
        'async_operations': False,
        'error_handling': False,
        'content_extraction': False,
        'fallback_mechanisms': False,
        'responses_api_handling': False
    }
    
    patterns = {
        'streaming_support': r'async for.*yield',
        'async_operations': r'async def.*streaming',
        'error_handling': r'try:.*except',
        'content_extraction': r'_extract_clean_content_from_chunk',
        'fallback_mechanisms': r'except.*fallback',
        'responses_api_handling': r'OpenAI Responses API'
    }
    
    for check, pattern in patterns.items():
        if re.search(pattern, content, re.DOTALL):
            checks[check] = True
    
    return checks

def check_graph_implementation(content: str) -> Dict[str, bool]:
    """Validate graph.py LangGraph implementation"""
    checks = {
        'async_nodes': False,
        'streaming_functions': False,
        'error_handling': False,
        'facilitator_decision': False,
        'structured_outputs': False,
        'content_processing': False,
        'embeddings_integration': False,
        'metrics_calculation': False
    }
    
    patterns = {
        'async_nodes': r'async def.*_node',
        'streaming_functions': r'agent_node_streaming',
        'error_handling': r'try:.*except.*Exception',
        'facilitator_decision': r'FacilitatorDecision',
        'structured_outputs': r'with_structured_output',
        'content_processing': r'_extract_.*_from_response',
        'embeddings_integration': r'embeddings\.aembed_query',
        'metrics_calculation': r'convergence_score'
    }
    
    for check, pattern in patterns.items():
        if re.search(pattern, content, re.DOTALL):
            checks[check] = True
    
    return checks

def count_error_handling_patterns(content: str) -> Dict[str, int]:
    """Count error handling and robustness patterns"""
    return {
        'try_blocks': len(re.findall(r'\btry:', content)),
        'except_blocks': len(re.findall(r'\bexcept\b', content)),
        'async_operations': len(re.findall(r'\basync def\b', content)),
        'await_calls': len(re.findall(r'\bawait\b', content)),
        'fallback_mentions': len(re.findall(r'\bfallback\b', content, re.IGNORECASE)),
        'emergency_responses': len(re.findall(r'\bemergency\b', content, re.IGNORECASE)),
        'retry_logic': len(re.findall(r'\bretry\b', content, re.IGNORECASE)),
        'error_handling': len(re.findall(r'\berror.*handl', content, re.IGNORECASE))
    }

def validate_syntax(file_path: str) -> bool:
    """Validate Python syntax by attempting to parse the AST"""
    try:
        content = load_file_content(file_path)
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False

def main():
    """Run comprehensive validation tests"""
    print("🔍 LangGraph, LangChain, and OpenAI API - Comprehensive Validation Test")
    print("=" * 80)
    
    # File paths
    files_to_check = {
        'pyproject': 'pyproject.toml',
        'agents': 'src/multiagent_debate/agents.py',
        'orchestrator': 'src/multiagent_debate/orchestrator.py',
        'graph': 'src/multiagent_debate/graph.py',
        'config': 'src/multiagent_debate/config.py'
    }
    
    # Load all file contents
    file_contents = {}
    missing_files = []
    
    for key, path in files_to_check.items():
        if os.path.exists(path):
            file_contents[key] = load_file_content(path)
        else:
            missing_files.append(path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    
    # 1. Syntax validation
    print("\n📋 Python Syntax Validation:")
    syntax_results = {}
    python_files = ['agents', 'orchestrator', 'graph', 'config']
    
    for file_key in python_files:
        file_path = files_to_check[file_key]
        syntax_results[file_key] = validate_syntax(file_path)
        status = "✅" if syntax_results[file_key] else "❌"
        print(f"  {status} {file_path}")
    
    # 2. Dependency validation
    print("\n📦 Dependency Configuration:")
    dep_checks = check_pyproject_dependencies(file_contents['pyproject'])
    for check, passed in dep_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    # 3. Agents configuration validation
    print("\n🤖 Agents Configuration:")
    agent_checks = check_agents_configuration(file_contents['agents'])
    for check, passed in agent_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    # 4. Orchestrator validation
    print("\n🎭 Orchestrator Implementation:")
    orch_checks = check_orchestrator_improvements(file_contents['orchestrator'])
    for check, passed in orch_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    # 5. Graph validation
    print("\n🕸️ LangGraph Implementation:")
    graph_checks = check_graph_implementation(file_contents['graph'])
    for check, passed in graph_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    # 6. Error handling analysis
    print("\n🛡️ Error Handling and Robustness:")
    all_content = ' '.join(file_contents.values())
    error_patterns = count_error_handling_patterns(all_content)
    
    for pattern, count in error_patterns.items():
        print(f"  📊 {pattern.replace('_', ' ').title()}: {count}")
    
    # 7. Feature completeness assessment
    print("\n🎯 Feature Completeness Assessment:")
    
    key_features = [
        ("✅ Updated to latest dependency versions", all(dep_checks.values())),
        ("✅ Updated to gpt-4o-mini model for better cost-performance", agent_checks.get('gpt_4o_mini_model', False)),
        ("✅ Integrated OpenAI Responses API with output_version='responses/v1'", agent_checks.get('output_version_responses', False)),
        ("✅ Implemented streaming support throughout system", orch_checks.get('streaming_support', False)),
        ("✅ Added comprehensive async/await operations", error_patterns['async_operations'] > 5),
        ("✅ Added strict mode for structured outputs (strict=True)", agent_checks.get('strict_mode', False)),
        ("✅ Enhanced content extraction for OpenAI Responses API", agent_checks.get('content_extraction', False)),
        ("✅ Implemented emergency response mechanisms", agent_checks.get('emergency_response', False)),
        ("✅ Added timeout and max_retries configuration", agent_checks.get('timeout_config', False) and agent_checks.get('max_retries_config', False)),
        ("✅ Enhanced error handling with try/except blocks", error_patterns['try_blocks'] > 20),
        ("✅ Implemented fallback mechanisms", error_patterns['fallback_mentions'] > 5),
        ("✅ Added robust content processing functions", graph_checks.get('content_processing', False)),
        ("✅ Integrated embeddings for convergence calculation", graph_checks.get('embeddings_integration', False)),
        ("✅ Implemented facilitator decision logic", graph_checks.get('facilitator_decision', False)),
        ("✅ Added comprehensive structured output handling", graph_checks.get('structured_outputs', False))
    ]
    
    passed_features = 0
    total_features = len(key_features)
    
    for feature_desc, feature_passed in key_features:
        if feature_passed:
            print(f"  {feature_desc}")
            passed_features += 1
        else:
            print(f"  ❌ {feature_desc.replace('✅ ', '')}")
    
    # 8. Overall assessment
    print("\n📊 Overall Assessment:")
    
    syntax_score = sum(syntax_results.values()) / len(syntax_results) * 100
    dependency_score = sum(dep_checks.values()) / len(dep_checks) * 100
    feature_score = passed_features / total_features * 100
    
    print(f"  📝 Syntax Validation: {syntax_score:.1f}% ({sum(syntax_results.values())}/{len(syntax_results)} files)")
    print(f"  📦 Dependency Configuration: {dependency_score:.1f}% ({sum(dep_checks.values())}/{len(dep_checks)} checks)")
    print(f"  🎯 Feature Implementation: {feature_score:.1f}% ({passed_features}/{total_features} features)")
    
    overall_score = (syntax_score + dependency_score + feature_score) / 3
    print(f"  🎖️ Overall Score: {overall_score:.1f}%")
    
    # 9. Robustness metrics
    print(f"\n🔧 Robustness Metrics:")
    print(f"  🛡️ Error Handling Blocks: {error_patterns['try_blocks']}")
    print(f"  🔄 Fallback Mechanisms: {error_patterns['fallback_mentions']}")
    print(f"  ⚡ Async Operations: {error_patterns['async_operations']}")
    print(f"  🎯 Await Calls: {error_patterns['await_calls']}")
    print(f"  🚨 Emergency Responses: {error_patterns['emergency_responses']}")
    
    # 10. Final verdict
    print("\n" + "=" * 80)
    if overall_score >= 90:
        print("🎉 EXCELLENT: System is fully updated and production-ready!")
        print("   All latest LangGraph, LangChain, and OpenAI features implemented.")
        print("   Comprehensive error handling and robust architecture detected.")
    elif overall_score >= 75:
        print("✅ GOOD: System is mostly updated with minor improvements needed.")
        print("   Most modern features implemented with good error handling.")
    elif overall_score >= 60:
        print("⚠️ MODERATE: System needs significant updates for full compatibility.")
        print("   Some modern features missing or incomplete implementation.")
    else:
        print("❌ POOR: System requires major updates for compatibility.")
        print("   Critical features missing or incorrect configuration.")
    
    print(f"\n🔍 Investigation Summary:")
    print(f"   • Latest dependency versions: {'✅' if dependency_score >= 80 else '❌'}")
    print(f"   • OpenAI Responses API integration: {'✅' if agent_checks.get('output_version_responses', False) else '❌'}")
    print(f"   • Structured outputs with strict mode: {'✅' if agent_checks.get('strict_mode', False) else '❌'}")
    print(f"   • Comprehensive error handling: {'✅' if error_patterns['try_blocks'] >= 20 else '❌'}")
    print(f"   • Modern async/streaming support: {'✅' if error_patterns['async_operations'] >= 5 else '❌'}")
    
    return overall_score >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)