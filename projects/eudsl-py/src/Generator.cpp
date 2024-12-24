// Copyright (c) 2024.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/LexicallyOrderedRecursiveASTVisitor.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"

#include <regex>

static llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::Required);

static llvm::cl::list<std::string>
    IncludeDirs("I", llvm::cl::desc("Directory of include files"),
                llvm::cl::value_desc("directory"), llvm::cl::Prefix);

static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::list<std::string>
    Namespace("namespace", llvm::cl::desc("Namespaces to generate from"),
              llvm::cl::CommaSeparated);

static bool filterInNamespace(const std::string &s) {
  if (Namespace.empty())
    return true;
  for (auto ns : Namespace)
    if (ns == s || ("::" + ns == s))
      return true;
  return false;
}

static std::string
getNBBindClassName(const std::string &qualifiedNameAsString) {
  std::string s = qualifiedNameAsString;
  s = std::regex_replace(s, std::regex(R"(\s+)"), "");
  s = std::regex_replace(s, std::regex("::"), "_");
  s = std::regex_replace(s, std::regex("[<|>]"), "__");
  s = std::regex_replace(s, std::regex(R"(\*)"), "___");
  s = std::regex_replace(s, std::regex(","), "____");
  return s;
}

static std::string getPyClassName(const std::string &qualifiedNameAsString) {
  std::string s = qualifiedNameAsString;
  s = std::regex_replace(s, std::regex(R"(\s+)"), "");
  s = std::regex_replace(s, std::regex(R"(\*)"), "");
  s = std::regex_replace(s, std::regex("<"), "[");
  s = std::regex_replace(s, std::regex(">"), "]");
  return s;
}

static std::string snakeCase(const std::string &name) {
  std::string s = name;
  s = std::regex_replace(s, std::regex(R"(([A-Z]+)([A-Z][a-z]))"), "$1_$2");
  s = std::regex_replace(s, std::regex(R"(([a-z\d])([A-Z]))"), "$1_$2");
  s = std::regex_replace(s, std::regex("-"), "_");
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

static clang::PrintingPolicy getPrintingPolicy(bool canonical = true) {
  clang::LangOptions lo;
  lo.CPlusPlus = true;
  clang::PrintingPolicy p(lo);
  // TODO(max): none of this really does anything except PrintCanonical
  // keep namespaces
  p.FullyQualifiedName = true;
  p.SuppressScope = false;
  p.SuppressInlineNamespace = false;
  p.PrintCanonicalTypes = canonical;
  p.Bool = true;

  return p;
}

// stolen from https://github.com/llvm/llvm-project/blob/99dddef340e566e9d303010f1219f7d7d6d37a11/clang/lib/Sema/SemaChecking.cpp#L7055
// Determines if the specified is a C++ class or struct containing
// a member with the specified name and kind (e.g. a CXXMethodDecl named
// "c_str()").
template <typename MemberKind>
static llvm::SmallPtrSet<MemberKind *, 1>
cxxRecordMembersNamed(clang::CXXMethodDecl *decl, clang::Sema &s) {
  llvm::SmallPtrSet<MemberKind *, 1> results;
  clang::LookupResult r(s, &s.Context.Idents.get(decl->getNameAsString()),
                        decl->getLocation(), clang::Sema::LookupMemberName);
  r.suppressDiagnostics();
  if (s.LookupQualifiedName(r, decl->getDeclContext()))
    for (clang::LookupResult::iterator i = r.begin(), e = r.end(); i != e;
         ++i) {
      clang::NamedDecl *namedDecl = (*i)->getUnderlyingDecl();
      if (MemberKind *fk = llvm::dyn_cast<MemberKind>(namedDecl))
        results.insert(fk);
    }
  return results;
}

static std::string sanitizeNameType(std::string name, int emptyIdx = 0) {
  if (name == "from")
    name = "from_";
  else if (name == "except")
    name = "except_";
  else if (name == "")
    name = std::string(emptyIdx + 1, '_');
  return name;
}

static bool emitClassMember(clang::CXXMethodDecl *decl,
                            clang::CompilerInstance &ci,
                            std::shared_ptr<llvm::ToolOutputFile> outputFile) {
  llvm::SmallVector<std::string> paramNames;
  llvm::SmallVector<std::string> paramTypes;
  for (unsigned i = 0; i < decl->getNumParams(); ++i) {
    clang::ParmVarDecl *param = decl->getParamDecl(i);
    std::string name = param->getNameAsString();
    auto t = param->getType();
    bool canonical = true;
    // TODO(max): this is dumb... (maybe there's a way to check where the typedef is defined...)
    if (t.getAsString().rfind("_t") != std::string::npos)
      canonical = false;
    paramTypes.push_back(
        sanitizeNameType(t.getAsString(getPrintingPolicy(canonical))));
    paramNames.push_back(sanitizeNameType(name, i));
  }

  const clang::CXXRecordDecl *parentRecord =
      llvm::cast<clang::CXXRecordDecl>(decl->getParent());
  std::string methodName = decl->getNameAsString();
  std::string className = parentRecord->getQualifiedNameAsString();
  auto isGetter = [](clang::CXXMethodDecl *d) {
    return d->getNameAsString().rfind("get", 0) == 0 && d->getNumParams() == 0;
  };

  std::string funcRef, nbFnName;
  // TODO(max): hasPointerRepresentation
  bool returnsPtr = decl->getType()->isPointerType(),
       returnsRef = decl->getType()->isReferenceType();

  if (llvm::isa<clang::CXXConstructorDecl>(decl)) {
    funcRef = llvm::formatv("nb::init<{0}>()", llvm::join(paramTypes, ", "));
  } else {
    llvm::SmallPtrSet<clang::CXXMethodDecl *, 1> overloads =
        cxxRecordMembersNamed<clang::CXXMethodDecl>(decl, ci.getSema());
    if (overloads.size() == 1) {
      funcRef = llvm::formatv("&{0}::{1}", className, methodName);
    } else {
      // emit a lambda body to disambiguate/break ties amongst overloads
      // TODO(max):: overloadimpl or whatever should work but it doesn't...
      std::string typedParamsStr;
      if (decl->getNumParams()) {
        llvm::SmallVector<std::string> typedParams =
            llvm::to_vector(llvm::map_range(
                llvm::zip(paramTypes, paramNames),
                [](std::tuple<std::string, std::string> item) -> std::string {
                  auto [t, n] = item;
                  return llvm::formatv("{0} {1}", t, n);
                }));
        typedParamsStr = llvm::join(typedParams, ", ");
      }
      // since we're emitting a body, we need to do std::move for some unique_ptrs
      llvm::SmallVector<std::string> newParamNames(paramNames);
      for (auto [idx, item] :
           llvm::enumerate(llvm::zip(paramTypes, newParamNames))) {
        // TODO(max): characterize this condition better...
        auto [t, n] = item;
        if ((t.rfind("std::unique_ptr") != std::string::npos &&
             t.back() != '&') ||
            t.rfind("&&") != std::string::npos)
          n = llvm::formatv("std::move({0})", n);
      }
      std::string newParamNamesStr = llvm::join(newParamNames, ", ");

      if (decl->isStatic()) {
        funcRef = llvm::formatv("[]({0}){{ return {1}{2}::{3}({4}); }",
                                typedParamsStr, returnsRef ? "&" : "",
                                className, methodName, newParamNamesStr);
      } else {
        if (decl->getNumParams())
          typedParamsStr = llvm::formatv("self, {0}", typedParamsStr);
        else
          typedParamsStr = "self";
        funcRef =
            llvm::formatv("[]({0}& {1}){{ return {2}self.{3}({4}); }",
                          className, typedParamsStr, returnsRef ? "&" : "",
                          decl->getNameAsString(), newParamNamesStr);
      }
    }

    nbFnName = methodName;
    if (decl->isOverloadedOperator()) {
      nbFnName = nbFnName;
    } else {
      // static method with non-static overloads that aren't also getters (getters are renamed already to break overlap)
      // see mlir::ElementsAttr
      if (overloads.size() > 1 &&
          llvm::any_of(overloads,
                       [isGetter](clang::CXXMethodDecl *m) {
                         return !m->isStatic() && !isGetter(m);
                       }) &&
          decl->isStatic() && !isGetter(decl)) {
        nbFnName += "_static";
      }

      // remove the redundant `get` because we're going to use def_prop below
      if (isGetter(decl) && nbFnName.rfind("get", 0) == 0)
        nbFnName = nbFnName.replace(0, 3, "");

      nbFnName = snakeCase(nbFnName);
    }
  }

  std::string paramNamesStr;
  if (decl->getNumParams()) {
    paramNames = llvm::to_vector(
        llvm::map_range(paramNames, [](std::string s) -> std::string {
          return llvm::formatv("\"{0}\"_a", snakeCase(s));
        }));
    paramNamesStr = ", " + llvm::join(paramNames, ", ");
  }

  std::string defStr = "def";
  if (decl->isStatic())
    defStr = "def_static";
  else if (isGetter(decl))
    defStr = "def_prop_ro";

  std::string refInternal;
  if (returnsPtr || returnsRef)
    refInternal = ", nb::rv_policy::reference_internal";

  if (!nbFnName.empty())
    nbFnName = llvm::formatv("\"{0}\", ", nbFnName);
  outputFile->os() << llvm::formatv(
      "{0}.{1}({2}{3}{4}{5});\n",
      getNBBindClassName(parentRecord->getQualifiedNameAsString()), defStr,
      nbFnName, funcRef, paramNamesStr, refInternal);

  return true;
}

static bool emitClass(clang::CXXRecordDecl *decl, clang::CompilerInstance &ci,
                      std::shared_ptr<llvm::ToolOutputFile> outputFile) {
  if (decl->isTemplated()) {
    clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
        decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                 clang::DiagnosticsEngine::Warning,
                                 "template classes not supported yet"));
    // have to force emit because after the fatal error, no more warnings will be emitted
    // https://github.com/llvm/llvm-project/blob/d74214cc8c03159e5d1f1168a09368cf3b23fd5f/clang/lib/Basic/DiagnosticIDs.cpp#L796
    (void)builder.setForceEmit();
    return false;
  }

  std::string scope = "m";
  const clang::DeclContext *declContext = decl->getDeclContext();
  if (declContext->isRecord()) {
    const clang::CXXRecordDecl *ctx =
        llvm::cast<clang::CXXRecordDecl>(declContext);
    scope = getNBBindClassName(ctx->getQualifiedNameAsString());
  }

  std::string className = decl->getQualifiedNameAsString();
  if (decl->getNumBases() > 1) {
    clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
        decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                 clang::DiagnosticsEngine::Warning,
                                 "multiple base classes not supported"));
    (void)builder.setForceEmit();
  } else if (decl->getNumBases() == 1) {
    clang::CXXBaseSpecifier baseClass = *decl->bases_begin();
    std::string baseName = baseClass.getType().getAsString(getPrintingPolicy());
    if (baseName.rfind("mlir::Op<", 0) == 0) {
      className = llvm::formatv("{0}, mlir::OpState", className);
    } else if (baseName.rfind("mlir::detail::StorageUserBase<", 0) == 0) {
      className = llvm::formatv("{0}, {1}", className,
                                llvm::StringRef{baseName}.split(",").first);
    } else {
      className = llvm::formatv("{0}, {1}", className, baseName);
    }
  }

  std::string autoVar = llvm::formatv(
      "auto {0}", getNBBindClassName(decl->getQualifiedNameAsString()));

  outputFile->os() << llvm::formatv("\n{0} = nb::class_<{1}>({2}, \"{3}\");\n",
                                    autoVar, className, scope,
                                    getPyClassName(decl->getNameAsString()));

  return true;
}

struct HackDeclContext : clang::DeclContext {
  bool islastDecl(clang::Decl *d) const { return d == LastDecl; }
  clang::Decl *getLastDecl() const { return LastDecl; }
};

struct ClassStructEnumVisitor
    : clang::LexicallyOrderedRecursiveASTVisitor<ClassStructEnumVisitor> {
  ClassStructEnumVisitor(clang::CompilerInstance &ci,
                         std::shared_ptr<llvm::ToolOutputFile> outputFile)
      : LexicallyOrderedRecursiveASTVisitor<ClassStructEnumVisitor>(
            ci.getSourceManager()),
        ci(ci), outputFile(outputFile),
        mc(clang::ItaniumMangleContext::create(ci.getASTContext(),
                                               ci.getDiagnostics())) {}

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *decl) {
    auto *encl = llvm::dyn_cast<clang::NamespaceDecl>(
        decl->getEnclosingNamespaceContext());
    if (!encl)
      return true;
    if (encl->isStdNamespace() || encl->isInStdNamespace())
      return true;
    if (!filterInNamespace(encl->getQualifiedNameAsString()))
      return true;
    if (!decl->isCompleteDefinition())
      return true;
    if (decl->getAccess() == clang::AS_private ||
        decl->getAccess() == clang::AS_protected)
      return true;

    if (decl->isClass() || decl->isStruct()) {
      if (emitClass(decl, ci, outputFile))
        visitedRecords.insert(decl);
    }

    return true;
  }

  bool VisitCXXMethodDecl(clang::CXXMethodDecl *decl) {
    if (visitedRecords.contains(decl->getParent()) &&
        (decl->getAccess() == clang::AS_public ||
         decl->getAccess() == clang::AS_none) &&
        !decl->isImplicit()) {
      emitClassMember(decl, ci, outputFile);
    }

    return true;
  }

  // implicit methods are the "last decls"
  bool shouldVisitImplicitCode() const { return true; }

  // TODO(max): this is a hack and not stable
  bool VisitDecl(clang::Decl *decl) {
    const clang::DeclContext *declContext = decl->getDeclContext();
    HackDeclContext *ctx =
        static_cast<HackDeclContext *>(decl->getDeclContext());
    if (declContext && declContext->isRecord()) {
      const clang::CXXRecordDecl *recordDecl =
          llvm::cast<clang::CXXRecordDecl>(declContext);
      if (visitedRecords.contains(recordDecl) && ctx->islastDecl(decl)) {
        outputFile->os() << "// " << recordDecl->getQualifiedNameAsString()
                         << "\n\n";
      }
    }
    return true;
  }

  clang::CompilerInstance &ci;
  std::shared_ptr<llvm::ToolOutputFile> outputFile;
  std::unique_ptr<clang::ItaniumMangleContext> mc;
  llvm::DenseSet<clang::CXXRecordDecl *> visitedRecords;
};

struct ClassStructEnumConsumer : clang::ASTConsumer {
  ClassStructEnumConsumer(
      clang::CompilerInstance &ci,
      const std::shared_ptr<llvm::ToolOutputFile> &outputFile)
      : visitor(ci, outputFile) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    if (!visitor.ci.hasSema())
      visitor.ci.createSema(clang::TU_Prefix,
                            /*CompletionConsumer*/ nullptr);
    visitor.TraverseDecl(context.getTranslationUnitDecl());
  }
  bool shouldSkipFunctionBody(clang::Decl *) override { return true; }
  ClassStructEnumVisitor visitor;
};

struct GenerateBindingsAction : clang::SyntaxOnlyAction {
  explicit GenerateBindingsAction(
      const std::shared_ptr<llvm::ToolOutputFile> &outputFile)
      : outputFile(outputFile) {}
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef inFile) override {
    // compiler.getPreprocessor().SetSuppressIncludeNotFoundError(true);
    compiler.getDiagnosticOpts().ShowLevel = true;
    compiler.getDiagnosticOpts().IgnoreWarnings = false;
    return std::make_unique<ClassStructEnumConsumer>(compiler, outputFile);
  }

  // PARSE_INCOMPLETE
  clang::TranslationUnitKind getTranslationUnitKind() override {
    return clang::TU_Prefix;
  }

  std::shared_ptr<llvm::ToolOutputFile> outputFile;
};

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }
  llvm::StringRef buffer = fileOrErr.get()->getBuffer();

  if (OutputFilename.empty()) {
    llvm::errs() << "-o outputfile must be set.\n";
    return -1;
  }

  std::error_code error;
  auto outputFile = std::make_shared<llvm::ToolOutputFile>(
      OutputFilename, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "cannot open output file '" + OutputFilename +
                        "': " + error.message();
    return -1;
  }
  std::vector<std::string> args{
      "-E",
      "-xc++",
      "-std=c++17",
      "-fsyntax-only",
      "-fdirectives-only",
      "-fkeep-system-includes",
      "-fdelayed-template-parsing",
      // "-nostdinc", "-nostdlibinc"
  };
  for (const auto &includeDir : IncludeDirs)
    args.emplace_back(llvm::formatv("-I{0}", includeDir));

  args.emplace_back(
      llvm::formatv("-I{0}", "/usr/lib/llvm-20/lib/clang/20/include"));
  args.emplace_back(llvm::formatv("-I{0}", "/usr/include/c++/12"));
  args.emplace_back(
      llvm::formatv("-I{0}", "/usr/include/x86_64-linux-gnu/c++/12"));
  args.emplace_back(llvm::formatv("-I{0}", "/usr/include"));
  args.emplace_back(llvm::formatv("-I{0}", "/usr/include/x86_64-linux-gnu"));

  if (!clang::tooling::runToolOnCodeWithArgs(
          std::make_unique<GenerateBindingsAction>(outputFile), buffer, args,
          InputFilename)) {
    llvm::errs() << "bindings generation failed.\n";
    // return -1;
  }

  outputFile->keep();

  return 0;
}