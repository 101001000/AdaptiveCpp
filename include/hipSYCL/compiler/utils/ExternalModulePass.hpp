/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_EXTERNAL_MODULE_PASS_HPP
#define HIPSYCL_EXTERNAL_MODULE_PASS_HPP

#include <iostream>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/IR/IRBuilder.h>


#include <vector>

namespace hipsycl {
namespace compiler {
namespace utils {

class ExternalModulePass : public llvm::PassInfoMixin<ExternalModulePass> {
public:
  ExternalModulePass() {}
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

    std::string BuiltinName = "__acpp_function_annotation_external_module";

    for (auto &F : M) {
      if (F.getName().contains(BuiltinName)) { //Find __acpp_function_annotation_external_module
        for (auto *U : F.users()) { // Get who's calling it (external_function)
          if (auto *CBB = llvm::dyn_cast<llvm::CallBase>(U)) { // Get calling instruction (call __acpp_function_annotation_external_module(%0,%1))
            llvm::Function* callerF = CBB->getParent()->getParent(); // Get external_function
            for (auto *UU : callerF->users()) {  // Get who is calling external_function (kernel or nd_range)
              if (auto *CB = llvm::dyn_cast<llvm::CallBase>(UU)) {  // Get calling instruction (call external_function)

                if(!callerF){
                  HIPSYCL_DEBUG_ERROR << "Caller function couldn't be retrieved \n";
                }
                llvm::Value *ArgPath  = CB->getArgOperand(0);
                llvm::Value *ArgFName = CB->getArgOperand(1);

                if(!ArgPath || !ArgFName){
                  HIPSYCL_DEBUG_ERROR << "Error retrieving operands \n";
                }

                CB->print(llvm::errs());

                llvm::GlobalVariable *GPath  = parse_global_from_arg(ArgPath);
                llvm::GlobalVariable *GFName = parse_global_from_arg(ArgFName);

                std::string path  = extract_string_from_global(M, std::string(GPath->getName()));
                std::string fname = extract_string_from_global(M, std::string(GFName->getName()));

                llvm::SMDiagnostic Err;

                std::unique_ptr<llvm::Module> MPayload = llvm::parseIRFile(path, Err, M.getContext());

                HIPSYCL_DEBUG_INFO << "Loading LLVM module in " << path << "\n";

                if(!MPayload){
                  HIPSYCL_DEBUG_ERROR << "Error loading LLVM Module \n";
                }
                if(llvm::Linker::linkModules(M, std::move(MPayload))){
                  HIPSYCL_DEBUG_ERROR << "Error linking LLVM Module \n";
                }

                llvm::Function* calleeF = find_function(M, fname);

                if(!calleeF){
                  HIPSYCL_DEBUG_ERROR << "Function " << fname << " not found in the linked module \n";
                }

                erase_function_body(callerF);
                insert_fcall(callerF, calleeF);
              }
            }
          }
        }
      }
    }
    return llvm::PreservedAnalyses::none();
  }

private:

  void erase_function_body(llvm::Function* F){
    std::vector<llvm::Instruction*> Is{};
    for(auto& BB : *F){
      for(auto& I : BB){
        Is.push_back(&I);
      }
    }
    for(llvm::Instruction* I : Is){
      I->eraseFromParent();
    }
  }

  llvm::Function* find_function(llvm::Module& M, std::string fname){
    for (auto& F : M) {
      if(F.getName().contains(fname)) {
        return &F;
      }
    }
    return nullptr;
  }

  void insert_fcall(llvm::Function* Caller, llvm::Function* Callee) {
    llvm::LLVMContext &Context = Caller->getContext();
    llvm::IRBuilder<> Builder(Context);
    llvm::BasicBlock &BB = Caller->getEntryBlock();
    Builder.SetInsertPoint(&BB, BB.getFirstInsertionPt());

    std::vector<llvm::Value*> Args;
    auto new_begin = std::next(Caller->arg_begin(), 2); // Skip path and fname arguments.
    for (auto it = new_begin; it != Caller->arg_end(); ++it) {
        Args.push_back(&*it);
    }

    llvm::CallInst *Call = Builder.CreateCall(Callee, Args);
    if (Caller->getReturnType()->isVoidTy()) {
        Builder.CreateRetVoid();
    } else {
        Builder.CreateRet(Call);
    }
  }

  std::string extract_string_from_global(const llvm::Module &M, std::string global_name) {
    for (auto &G : M.globals()) {
      if (std::string(G.getName()) == global_name) {
        auto da = llvm::dyn_cast<llvm::ConstantDataArray>(G.getInitializer());
        if (da) {
          auto cs = da->getAsCString();
          return std::string(cs);
        }
      }
    }
    HIPSYCL_DEBUG_ERROR << "String could not be retrieved from " << global_name << "\n";
    return "no";
  }
  llvm::GlobalVariable* parse_global_from_arg(llvm::Value* Arg){
    if (auto *ConstExpr = llvm::dyn_cast<llvm::ConstantExpr>(Arg)) {
      if (ConstExpr->getOpcode() == llvm::Instruction::AddrSpaceCast) {
        llvm::Value *Operand = ConstExpr->getOperand(0);
        if (auto *GlobalVar = llvm::dyn_cast<llvm::GlobalVariable>(Operand)) {
          return GlobalVar;
        }
      }
    }
    HIPSYCL_DEBUG_ERROR << "Global could not be retrieved from arg" << std::string(Arg->getName()) << "\n";
    return nullptr;
  }
};

} // namespace utils
} // namespace compiler
} // namespace hipsycl

#endif
