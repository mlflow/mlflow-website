import React from "react";

import { Button } from "../Button/Button";

interface Props {
  code: string;
}

export const CopyCommand = ({ code }: Props) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="bg-gray-100 backdrop-blur-[24px] rounded-[32px] p-3 border border-gray-200 shadow-lg">
        <div className="bg-white rounded-[20px] p-2 pl-5 border border-gray-300">
          <div className="flex items-center justify-between gap-20">
            <div className="flex-1">
              <div className="text-black !font-[DM_Mono]">{code}</div>
            </div>
            <Button variant="secondary" size="medium" onClick={handleCopy}>
              Copy to begin
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
